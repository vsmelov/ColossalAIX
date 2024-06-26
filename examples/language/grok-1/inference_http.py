"""
curl -X POST "http://51.107.21.198:40670/inference/" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hi! How are you?", "max_new_tokens": 100}'

curl -X POST "http://51.107.21.196:40007/inference/" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hi! How are you?", "max_new_tokens": 100}'

"""

import time
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from grok1_policy import Grok1ForCausalLMPolicy
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_default_parser, inference

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.cluster import DistCoordinator
from colossalai.lazy import LazyInitContext
from colossalai.utils import get_current_device
import torch.distributed as dist
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s',
)

# dist.init_process_group("nccl")

logging.info(f'Hello World!')

parser = get_default_parser()
args = parser.parse_args()

app = FastAPI()


from transformers import StoppingCriteria

STOP_SEQUENCE = "="*10

class MyStoppingCriteria(StoppingCriteria):
    def __init__(self, target_sequence, prompt):
        self.target_sequence = target_sequence
        self.prompt = prompt

    def __call__(self, input_ids, scores, **kwargs):
        # Get the generated text as a string
        generated_text = tokenizer.decode(input_ids[0])
        generated_text = generated_text.replace(self.prompt,'')
        # Check if the target sequence appears in the generated text
        if self.target_sequence in generated_text:
            return True  # Stop generation

        return False  # Continue generation

    def __len__(self):
        return 1

    def __iter__(self):
        yield self


class TextRequest(BaseModel):
    text: str
    max_new_tokens: int = 100
    do_sample: bool = True
    temperature: float = 0.3
    top_k: int = 20  # 50
    top_p: float = 0.5  # 0.95


import threading
model_lock = threading.Lock()


@app.post("/inference/")
def do_inference(request: TextRequest):
    logging.info(f"Received request: {request}")
    with model_lock:
        start_time = time.time()
        try:
            create_tasks(request)

            output = inference(
                model.unwrap(),
                tokenizer,
                request.text,
                max_new_tokens=request.max_new_tokens,
                do_sample=request.do_sample,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                stopping_criteria=MyStoppingCriteria(STOP_SEQUENCE, request.text)
            )
            response = tokenizer.decode(output)
            duration = time.time() - start_time
            logging.info(f'Inference took {duration:.2f} seconds')
            logging.info(f"Response: {response}")
            return {"response": response, 'duration': duration}
        except Exception as e:
            logging.exception(f'Error occurred: {e}')
            raise HTTPException(status_code=500, detail=str(e))


def create_tasks(request: TextRequest):
    logging.info(f"Creating tasks for {request}")
    for i in range(1, 8):
        create_task(i, request)


def task_f(worker):
    return f'worker-{worker}.task.json'


def remove_tasks():
    for i in range(1, 8):
        import os
        f = task_f(i)
        if os.path.exists(f):
            logging.info(f"Removing task file: {f}")
            os.remove(f)


remove_tasks()


def create_task(worker, request: TextRequest):
    import json
    with open(task_f(worker), 'w') as f:
        json.dump(request.dict(), f)


if __name__ == "__main__":
    logging.info(f"Starting MAIN, {args=}")
    start = time.time()

    colossalai.launch_from_torch({})
    logging.info(f"colossalai.launch_from_torch()")

    coordinator = DistCoordinator()
    logging.info(f"DistCoordinator()")

    plugin = HybridParallelPlugin(
        tp_size=coordinator.world_size,
        pp_size=1,
        precision="bf16",
        parallel_output=False,
        custom_policy=Grok1ForCausalLMPolicy(),
    )
    logging.info(f"HybridParallelPlugin()")

    booster = Booster(plugin=plugin)
    logging.info(f"Booster()")

    torch.set_default_dtype(torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained, trust_remote_code=True)
    logging.info(f"AutoTokenizer()")

    with LazyInitContext(default_device=get_current_device()):
        model = AutoModelForCausalLM.from_pretrained(
            args.pretrained, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        logging.info(f"AutoModelForCausalLM()")
    model, *_ = booster.boost(model)
    logging.info(f"Booster.boost()")
    model.eval()
    logging.info(f"model.eval()")
    init_time = time.time() - start

    logging.info(f"Model initialized in {init_time:.2f} seconds, rank={dist.get_rank()}")

    # Start FastAPI only on the master node
    if dist.get_rank() == 0:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        fname = f'worker-{dist.get_rank()}.task.json'
        while True:
            # this is the worst synchronization method ever

            # if file exist
            import os
            if not os.path.exists(fname):
                time.sleep(0.3)
                continue
            with open(f'worker-{dist.get_rank()}.task.json', 'r') as f:
                import json
                task = json.load(f)
                logging.info(f"Worker {dist.get_rank()} received task: {task}")
                output = inference(
                    model.unwrap(),
                    tokenizer,
                    task['text'],
                    max_new_tokens=task['max_new_tokens'],
                    do_sample=task['do_sample'],
                    temperature=task['temperature'],
                    top_k=task['top_k'],
                    top_p=task['top_p'],
                    stopping_criteria=MyStoppingCriteria(STOP_SEQUENCE, task['text']),
                )
            # remove the task file
            os.remove(f'worker-{dist.get_rank()}.task.json')
