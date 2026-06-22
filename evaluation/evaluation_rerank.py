import asyncio
import json
import statistics
import time
from pathlib import Path

from apps.rag.bm25 import search_bm25
from apps.rag.fusion import merge_candidates
from apps.rag.reranker import rerank_chunks
from apps.rag.retrieval import retrieve_chunk
from config.db_config import AsyncSessionLocal, engine
from config.qdrant_config import client


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.jsonl"
RESULT_DIR = BASE_DIR / "results" / "v2_42q"

USER_ID = 1
CANDIDATE_K = 30
FINAL_K = 5
SCORE_THRESHOLD = 0.0
FINAL_EVALUATE_AT = [1, 3, 5]


def load_dataset() -> list[dict]:
    rows = []

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"数据集第 {line_number} 行不是合法 JSON"
                ) from exc

            rows.append(row)

    return rows


def calculate_rank_metrics(
    retrieved_ids: list[int],
    relevant_ids: list[int],
    k: int,
) -> dict:
    predictions = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    matched = relevant_set.intersection(predictions)

    reciprocal_rank = 0.0
    for rank, chunk_id in enumerate(predictions, start=1):
        if chunk_id in relevant_set:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "hit": 1.0 if matched else 0.0,
        "recall": (
            len(matched) / len(relevant_set)
            if relevant_set
            else 0.0
        ),
        "reciprocal_rank": reciprocal_rank,
    }


def calculate_candidate_metrics(
    candidate_ids: list[int],
    relevant_ids: list[int],
) -> dict:
    candidate_set = set(candidate_ids)
    relevant_set = set(relevant_ids)
    matched = relevant_set.intersection(candidate_set)

    return {
        "hit": 1.0 if matched else 0.0,
        "recall": (
            len(matched) / len(relevant_set)
            if relevant_set
            else 0.0
        ),
        "matched_chunk_ids": sorted(matched),
    }


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0


    sorted_values = sorted(values)
    index = int((len(sorted_values) - 1) * ratio)
    return sorted_values[index]


def mean(values) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def summarize_group(results: list[dict]) -> dict:
    if not results:
        return {
            "question_count": 0,
            "candidate": {},
            "rerank": {},
        }

    summary = {
        "question_count": len(results),
        "candidate": {
            "hit_rate": round(
                mean(
                    result["candidate_metrics"]["hit"]
                    for result in results
                ),
                4,
            ),
            "recall": round(
                mean(
                    result["candidate_metrics"]["recall"]
                    for result in results
                ),
                4,
            ),
            "average_count": round(
                mean(result["candidate_count"] for result in results),
                2,
            ),
        },
        "rerank": {},
    }

    for k in FINAL_EVALUATE_AT:
        key = f"@{k}"
        summary["rerank"][key] = {
            "hit_rate": round(
                mean(
                    result["rerank_metrics"][key]["hit"]
                    for result in results
                ),
                4,
            ),
            "recall": round(
                mean(
                    result["rerank_metrics"][key]["recall"]
                    for result in results
                ),
                4,
            ),
            "mrr": round(
                mean(
                    result["rerank_metrics"][key][
                        "reciprocal_rank"
                    ]
                    for result in results
                ),
                4,
            ),
        }

    return summary


async def evaluate() -> None:
    dataset = load_dataset()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    answerable_rows = [
        row for row in dataset
        if row["answerable"]
    ]

    results = []
    retrieval_latencies = []
    rerank_latencies = []
    total_latencies = []

    async with AsyncSessionLocal() as db:
        for index, row in enumerate(answerable_rows, start=1):
            relevant_ids = row["relevant_chunk_ids"]

            if not relevant_ids:
                raise ValueError(
                    f"{row['id']} 没有填写 relevant_chunk_ids"
                )

            total_started_at = time.perf_counter()
            retrieval_started_at = time.perf_counter()

            vector_results = await retrieve_chunk(
                user_query=row["question"],
                user_id=USER_ID,
                document_id=row["document_id"],
                db=db,
                top_k=CANDIDATE_K,
                score_threshold=SCORE_THRESHOLD,
            )

            bm25_results = await search_bm25(
                query=row["question"],
                user_id=USER_ID,
                document_id=row["document_id"],
                db=db,
                top_k=CANDIDATE_K,
            )

            candidates = merge_candidates(
                vector_results,
                bm25_results,
            )

            retrieval_latency_ms = (
                time.perf_counter() - retrieval_started_at
            ) * 1000

            rerank_started_at = time.perf_counter()

            reranked = await rerank_chunks(
                query=row["question"],
                items=candidates,
                top_n=FINAL_K,
            )

            rerank_latency_ms = (
                time.perf_counter() - rerank_started_at
            ) * 1000
            total_latency_ms = (
                time.perf_counter() - total_started_at
            ) * 1000

            retrieval_latencies.append(retrieval_latency_ms)
            rerank_latencies.append(rerank_latency_ms)
            total_latencies.append(total_latency_ms)

            vector_ids = [
                item.metadata.chunk_index
                for item in vector_results
            ]
            bm25_ids = [
                item.metadata.chunk_index
                for item in bm25_results
            ]
            candidate_ids = [
                item.metadata.chunk_index
                for item in candidates
            ]
            reranked_ids = [
                item.metadata.chunk_index
                for item in reranked
            ]

            candidate_metrics = calculate_candidate_metrics(
                candidate_ids,
                relevant_ids,
            )
            rerank_metrics = {
                f"@{k}": calculate_rank_metrics(
                    reranked_ids,
                    relevant_ids,
                    k,
                )
                for k in FINAL_EVALUATE_AT
            }

            result = {
                "id": row["id"],
                "category": row["category"],
                "source_question_id": row.get(
                    "source_question_id"
                ),
                "question": row["question"],
                "document_id": row["document_id"],
                "relevant_chunk_ids": relevant_ids,
                "vector_chunk_ids": vector_ids,
                "bm25_chunk_ids": bm25_ids,
                "candidate_chunk_ids": candidate_ids,
                "candidate_count": len(candidate_ids),
                "candidate_metrics": candidate_metrics,
                "reranked_chunk_ids": reranked_ids,
                "reranked_scores": [
                    item.score for item in reranked
                ],
                "rerank_metrics": rerank_metrics,
                "latency_ms": {
                    "retrieval": round(
                        retrieval_latency_ms,
                        2,
                    ),
                    "rerank": round(rerank_latency_ms, 2),
                    "total": round(total_latency_ms, 2),
                },
            }
            results.append(result)

            print(
                f"[{index}/{len(answerable_rows)}] "
                f"{row['id']} "
                f"CandidateHit="
                f"{candidate_metrics['hit']:.0f} "
                f"RerankHit@5="
                f"{rerank_metrics['@5']['hit']:.0f} "
                f"MRR@5="
                f"{rerank_metrics['@5']['reciprocal_rank']:.3f} "
                f"{total_latency_ms:.0f}ms"
            )

    original_results = [
        result for result in results
        if result["category"] in {"fact", "synthesis"}
    ]
    paraphrase_results = [
        result for result in results
        if result["category"] == "paraphrase"
    ]

    summary = {
        "config": {
            "user_id": USER_ID,
            "vector_candidate_k": CANDIDATE_K,
            "bm25_candidate_k": CANDIDATE_K,
            "final_k": FINAL_K,
            "score_threshold": SCORE_THRESHOLD,
            "question_count": len(results),
        },
        "groups": {
            "all": summarize_group(results),
            "original": summarize_group(original_results),
            "paraphrase": summarize_group(
                paraphrase_results
            ),
        },
        "latency_ms": {
            "retrieval": {
                "average": round(
                    mean(retrieval_latencies),
                    2,
                ),
                "p50": round(
                    percentile(retrieval_latencies, 0.50),
                    2,
                ),
                "p95": round(
                    percentile(retrieval_latencies, 0.95),
                    2,
                ),
            },
            "rerank": {
                "average": round(mean(rerank_latencies), 2),
                "p50": round(
                    percentile(rerank_latencies, 0.50),
                    2,
                ),
                "p95": round(
                    percentile(rerank_latencies, 0.95),
                    2,
                ),
            },
            "total": {
                "average": round(mean(total_latencies), 2),
                "p50": round(
                    percentile(total_latencies, 0.50),
                    2,
                ),
                "p95": round(
                    percentile(total_latencies, 0.95),
                    2,
                ),
            },
        },
    }

    details_path = (
        RESULT_DIR
        / f"union_{CANDIDATE_K}_rerank_{FINAL_K}_details.json"
    )
    summary_path = (
        RESULT_DIR
        / f"union_{CANDIDATE_K}_rerank_{FINAL_K}_summary.json"
    )

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
