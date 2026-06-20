import os
from dotenv import load_dotenv
import httpx

from apps.rag.schemas import RetrieveItem

load_dotenv()

RERANK_URL = (
    "https://dashscope.aliyuncs.com/"
    "compatible-api/v1/reranks"
)


async def rerank_chunks(
    query: str,
    items: list[RetrieveItem],
    top_n: int = 5,
) -> list[RetrieveItem]:
    if not items:
        return []
    payload = {
        "model": os.getenv("RERANK_MODEL", "qwen3-rerank"),
        "query": query,
        "documents": [item.content for item in items],
        "top_n": min(top_n, len(items)),
        "instruct": (
            "Given a user question, retrieve passages "
            "that directly answer the question."
        ),
    }

    headers = {
        "Authorization": f"Bearer {os.getenv('API_KEY')}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            RERANK_URL,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        reranked = []
        results = response.json()["results"]
        for result in results:
            original_item = items[result["index"]]
            reranked.append(original_item.model_copy(update={"score": result["relevance_score"]}))
    return reranked