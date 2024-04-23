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

app = FastAPI()

# Define a request model
class TextRequest(BaseModel):
    text: str
    max_new_tokens: int = 100


@app.post("/inference/")
def do_inference(request: TextRequest):
    try:
        output = inference(
            model.unwrap(),
            tokenizer,
            request.text,
            max_new_tokens=request.max_new_tokens
        )
        return {"response": tokenizer.decode(output)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Initialize the distributed environment
    dist.init_process_group("nccl")

    parser = get_default_parser()
    args = parser.parse_args()

    colossalai.launch_from_torch({})
    coordinator = DistCoordinator()
    plugin = HybridParallelPlugin(
        tp_size=coordinator.world_size,
        pp_size=1,
        precision="bf16",
        parallel_output=False,
        custom_policy=Grok1ForCausalLMPolicy(),
    )
    booster = Booster(plugin=plugin)
    torch.set_default_dtype(torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained, trust_remote_code=True)

    with LazyInitContext(default_device=get_current_device()):
        model = AutoModelForCausalLM.from_pretrained(
            args.pretrained, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
    model, *_ = booster.boost(model)
    model.eval()

    # Start FastAPI only on the master node
    if dist.get_rank() == 0:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # Other processes do computational work
        dist.barrier()  # This could be used to synchronize processes if needed
