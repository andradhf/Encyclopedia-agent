"""Simple RAGAS evaluation for the encyclopedia RAG agent.

Runs the golden question set in dataset.json through the agent, harvests
(question, answer, contexts) via harness.run_question, and scores the run
with reference-free RAGAS metrics (no ground-truth answers required).

Usage:
    python -m encyclopedia_agent.Service.Evaluation.run_eval
    python -m encyclopedia_agent.Service.Evaluation.run_eval --limit 2
"""

import argparse
import datetime as dt
import json
import os
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.memory import InMemorySaver
from loguru import logger
from sentence_transformers import SentenceTransformer

from ._ragas_compat import patch_ragas_imports

patch_ragas_imports()

from ragas import EvaluationDataset, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig  # noqa: E402

from ..EncyclopediaAgent.graph import build_agent  # noqa: E402
from .harness import run_question  # noqa: E402

DATASET_PATH = Path(__file__).parent / "dataset.json"
OUTPUT_DIR = Path(__file__).parent / "output"

AGENT_MODEL = "google_genai:gemini-3.1-flash-lite"  # keep in sync with graph.py's build_agent()
JUDGE_MODEL = os.environ.get("RAGAS_JUDGE_MODEL", AGENT_MODEL)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class _SentenceTransformerEmbeddings(Embeddings):
    """Minimal langchain Embeddings adapter around the local bge model.

    ragas 0.4.3's own HuggingFace embeddings classes are unusable here: the
    new `HuggingFaceEmbeddings` lacks the .embed_query() the legacy
    ResponseRelevancy metric calls, and the legacy `HuggingfaceEmbeddings` is
    itself abstract (missing async methods). langchain_core.embeddings.Embeddings
    provides a default executor-based async implementation, so implementing
    just the two sync methods here is enough for LangchainEmbeddingsWrapper.
    """

    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def load_dataset(limit: int | None) -> list[dict]:
    questions = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if limit:
        questions = questions[:limit]
    return questions


def collect_samples(questions: list[dict]) -> list[dict]:
    agent = build_agent(checkpointer=InMemorySaver())
    samples = []
    for item in questions:
        logger.info("Running {}: {!r}", item["id"], item["question"])
        result = run_question(agent, item["question"])
        if result["error"]:
            logger.warning("Skipping {} due to error: {}", item["id"], result["error"])
            continue
        if not result["contexts"]:
            logger.warning("{} retrieved no contexts", item["id"])
        samples.append(
            {
                "user_input": result["question"],
                "response": result["answer"],
                "retrieved_contexts": result["contexts"],
            }
        )
    return samples


def run_ragas(samples: list[dict]):
    # bypass_n=True: ChatGoogleGenerativeAI exposes an `n` attribute, so ragas
    # tries to request multiple candidates in one call (used by
    # ResponseRelevancy). Gemini rejects that ("Multiple candidates is not
    # enabled for this model"), so force ragas to issue n separate calls
    # instead.
    judge_llm = LangchainLLMWrapper(init_chat_model(model=JUDGE_MODEL), bypass_n=True)
    embeddings = LangchainEmbeddingsWrapper(_SentenceTransformerEmbeddings(EMBEDDING_MODEL))

    dataset = EvaluationDataset.from_list(samples)
    return evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            # strictness=1: default of 3 asks the judge to generate 3
            # candidate questions per sample (3 LLM calls); free-tier Gemini
            # rate limits can't absorb that on top of everything else.
            ResponseRelevancy(strictness=1),
            LLMContextPrecisionWithoutReference(),
        ],
        llm=judge_llm,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=1, timeout=300),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None, help="Only run the first N questions"
    )
    args = parser.parse_args()

    questions = load_dataset(args.limit)
    logger.info("Loaded {} questions", len(questions))

    samples = collect_samples(questions)
    if not samples:
        logger.error("No samples collected, aborting evaluation")
        return

    result = run_ragas(samples)

    print("\n=== RAGAS scores (mean) ===")
    print(result)

    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"eval_{timestamp}.csv"
    result.to_pandas().to_csv(csv_path, index=False)
    print(f"\nPer-sample scores written to {csv_path}")


if __name__ == "__main__":
    main()
