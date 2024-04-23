"""
pip install fastapi uvicorn
"""

from fastapi import FastAPI
from pydantic import BaseModel
import time

app = FastAPI()


# Define a request model
class TextRequest(BaseModel):
    text: str
    max_new_tokens: int = 100


@app.post("/inference/")
async def process_request(request: TextRequest):
    start_time = time.time()

    response_text = f"Processed: {request.text}"
    duration = time.time() - start_time

    return {
        "response": response_text,
        "duration": duration
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
