import json
from typing import List, Tuple

import torch
import faiss

from model import tokenizer, nn_model


def load_json_data() -> List[str]:
    # Load data
    with open("./data/facts.json") as f:
        data = json.load(f)
        terms: List[str] = [
            "The definition of " + term["term"] + " is " + term["definition"]
            for term in data["terms"]
        ]
        facts: List[str] = data["facts"]
        return terms + facts
    return []


def get_embeddings(texts: List[str]) -> torch.Tensor:
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
    with torch.no_grad():
        outputs = nn_model(**inputs)
        embeddings = outputs.logits[:, 0, :]
    return embeddings


def create_cpu_index() -> Tuple[faiss.Index, List[str]]:
    data = load_json_data()
    embeddings = get_embeddings(data)
    embeddings_np = embeddings.cpu().numpy()
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    return index, data


cpu_index, rag_data = create_cpu_index()

def rag_retrieve(query: str, top_k: int = 1) -> List[str]:
    global cpu_index, rag_data
    query_embedding = get_embeddings([query])
    query_embedding = query_embedding.cpu().numpy()
    distances, indices = cpu_index.search(query_embedding.reshape(1, -1), top_k)
    return [rag_data[i] for i in indices.flatten()]
