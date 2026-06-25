import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel
from ragas import Dataset, experiment

from apps.rag.workflow.context import RAGRuntimeContext, RetrievalConfig
from apps.rag.workflow.graph import create_rag_graph
from config.db_config import AsyncSessionLocal


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.jsonl"
RESULT_DIR = BASE_DIR / "results" / "ragas"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

USER_ID = 1
DOCUMENT_ID = 2


class RagasInputRow(BaseModel):
    question: str
    reference: str | None = None
    document_id: int | None = None


class RagasExperimentRow(BaseModel):
    question: str
    answer: str
    contexts: list[str]
    reference: str | None = None
    context_count: int


def load_rows() -> list[RagasInputRow]:
    rows = []

    with DATASET_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            raw = json.loads(line)

            rows.append(
                RagasInputRow(
                    question=raw.get("question")
                    or raw.get("user_input")
                    or raw.get("query"),
                    reference=raw.get("reference")
                    or raw.get("answer")
                    or raw.get("ground_truth"),
                    document_id=raw.get("document_id"),
                )
            )

    return rows


@experiment(RagasExperimentRow, name_prefix="agentic-rag")
async def run_rag_experiment(row: RagasInputRow) -> RagasExperimentRow:
    graph = create_rag_graph(None)

    async with AsyncSessionLocal() as db:
        result = await graph.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": row.question,
                    }
                ],
                "original_query": row.question,
                "search_query": row.question,
                "rewrite_count": 0,
            },
            context=RAGRuntimeContext(
                user_id=USER_ID,
                document_id=row.document_id or DOCUMENT_ID,
                db=db,
                retrieval=RetrievalConfig(
                    candidate_k=30,
                    final_k=5,
                    score_threshold=0.0,
                ),
            ),
        )

    candidates = result.get("candidates", [])
    contexts = [item.content for item in candidates]

    return RagasExperimentRow(
        question=row.question,
        answer=result.get("answer", ""),
        contexts=contexts,
        reference=row.reference,
        context_count=len(contexts),
    )


async def main():
    rows = load_rows()

    dataset = Dataset(
        name="agentic_rag_eval",
        backend="local/csv",
        data_model=RagasInputRow,
        data=rows,
        root_dir=str(RESULT_DIR),
    )

    result = await run_rag_experiment.arun(
        dataset,
        name="ragas_agentic_rag_baseline",
        backend="local/csv",
    )

    print(f"experiment saved: {result.name}")
    print(f"row count: {len(result)}")


if __name__ == "__main__":
    asyncio.run(main())