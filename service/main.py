from typing import Tuple
import json

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from model import inpput_to_output
from rag import rag_retrieve


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if torch.cuda.is_available():
    print("CUDA is available!")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")


def get_prompts() -> Tuple[str, str]:
    with open("./data/prompt.json") as f:
        data = json.load(f)
        return data["prefix"], data["suffix"]
    return "", ""


def prepare_input_text(question: str) -> str:
    input = ""
    # First RAG
    retrieved_texts = rag_retrieve(question, top_k=1)
    input += "Given the following facts: " + ".".join(retrieved_texts)

    # Then question part
    prefix, suffix = get_prompts()
    input += prefix + " " + question + " " + suffix

    return input


def question_to_answer(question: str):
    input = prepare_input_text(question)
    output = inpput_to_output(input)
    return output


class ChatRequest(BaseModel):
    input: str


@app.post("/chat")
async def generate_text(request: ChatRequest) -> str:
    try:
        input = prepare_input_text(request.input)
        return inpput_to_output(input)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve static files from React build after all API routes
app.mount("/", StaticFiles(directory="../web/build", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8889)
