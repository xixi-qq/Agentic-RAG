import asyncio
import json
import statistics
import time
from pathlib import Path
from apps.rag.retrieval import retrieve_chunk
from config.db_config import AsyncSessionLocal, engine
from config.qdrant_config import client

DATASET_PATH = Path("dataset.jsonl")
RESULT_DIR = Path("results")

USER_ID = 1
TOP_K = 30
SCORE_THRESHOLD = 0.0
EVALUATE_AT = [1, 3, 5, 10, 15,30]


def load_dataset() -> list[dict]:
    rows = []
    with DATASET_PATH.open("r",encoding="utf-8") as f:
        for i,line in enumerate(f):
            if not line.strip():
                 continue
            try:
                row = json.loads( line)
            except json.JSONDecodeError:
                print(f"Error decoding line {i}")
            rows.append(row)
    return rows



def calculate_metrics(retrieve_ids: list[int], relevant_ids: list[int],k: int) -> dict:
    predictions = retrieve_ids[:k]
    relevant_set = set(relevant_ids)
    matched = relevant_set.intersection( predictions)
    hit = 1.0 if matched else 0.0
    recall = len(matched) / len(relevant_set) if relevant_set else 0.0
    reciprocal_rank = 0.0
    for i, prediction in enumerate(predictions,start=1):
        if prediction in relevant_set:
            reciprocal_rank = 1.0 / i
            break
    return  {
        "hit": hit,
        "recall": recall,
        "reciprocal_rank": reciprocal_rank,

    }


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0

    sorted_values = sorted(values)
    index = int((len(sorted_values) - 1) * ratio)

    return sorted_values[index]



async def evaluate():
    dataset = load_dataset()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    answerable_rows = [row for row in dataset if row["answerable"]]
    results = []
    latencies = []
    async with AsyncSessionLocal() as db:
        for i, row in enumerate(answerable_rows,start=1):
            relevant_ids = row["relevant_chunk_ids"]
            if not relevant_ids:
                raise ValueError(f"{row['id']} 没有填写 relevant_chunk_ids")

            started_at = time.perf_counter()

            retrieved = await retrieve_chunk(
                user_query=row["question"],
                user_id=USER_ID,
                document_id=row["document_id"],
                db=db,
                top_k=TOP_K,
                score_threshold=SCORE_THRESHOLD,
            )
            latency_ms = (
                                 time.perf_counter() - started_at
                         ) * 1000

            latencies.append(latency_ms)

            retrieved_ids = [item.metadata.chunk_index for item in retrieved]
            scores = [item.score for item in retrieved]
            metrics = {
                f"@{k}": calculate_metrics(
                    retrieved_ids,
                    relevant_ids,
                    k,
                )
                for k in EVALUATE_AT
            }

            result = {
                "id": row["id"],
                "question": row["question"],
                "document_id": row["document_id"],
                "relevant_chunk_ids": relevant_ids,
                "retrieved_chunk_ids": retrieved_ids,
                "scores": scores,
                "latency_ms": round(latency_ms, 2),
                "metrics": metrics,
            }

            results.append(result)

            print(
                f"[{i}/{len(answerable_rows)}] "
                f"{row['id']} "
                f"Hit@5={metrics['@5']['hit']:.0f} "
                f"MRR@5={metrics['@5']['reciprocal_rank']:.3f} "
                f"{latency_ms:.0f}ms"
            )

    summary = {
    "config": {
        "user_id": USER_ID,
        "top_k": TOP_K,
        "score_threshold": SCORE_THRESHOLD,
        "question_count": len(answerable_rows),
    },
    "metrics": {},
    "latency_ms": {
        "average": round(statistics.mean(latencies), 2),
        "p50": round(percentile(latencies, 0.50), 2),
        "p95": round(percentile(latencies, 0.95), 2),
    },
}
    for k in EVALUATE_AT:
        key = f"@{k}"
        summary["metrics"][key] = {
            "hit_rate": round(statistics.mean(result["metrics"][key]["hit"] for result in results),4),
            "recall": round(statistics.mean(result["metrics"][key]["recall"] for result in results),4),
            "mrr": round(statistics.mean(result["metrics"][key]["reciprocal_rank"] for result in results),4),

        }

    details_path = RESULT_DIR / f"{summary["config"]["top_k"]}_{summary["config"]["score_threshold"]}_vector_baseline_details.json"
    summary_path = RESULT_DIR / f"{summary["config"]["top_k"]}_{summary["config"]["score_threshold"]}_vector_baseline_summary.json"

    details_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n评估完成：")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n详细结果：{details_path}")
    print(f"汇总结果：{summary_path}")



async def main() -> None:
    try:
        await evaluate()
    finally:
        await client.close()
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())






