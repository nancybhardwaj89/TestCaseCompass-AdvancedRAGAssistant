import json
import os
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mistralai import MistralAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    answer_correctness,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

load_dotenv()

DATASET_PATH = Path("datasets/golden_dataset.json")

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing in the .env file.")
if not os.getenv("MISTRAL_API_KEY"):
    raise ValueError("MISTRAL_API_KEY is missing in the .env file.")

with open(DATASET_PATH, "r", encoding="utf-8") as file:
    raw_dataset = json.load(file)

evaluator_llm = LangchainLLMWrapper(
    ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=4096)
)
evaluator_embeddings = LangchainEmbeddingsWrapper(
    MistralAIEmbeddings(model="mistral-embed")
)

metrics = {
    "faithfulness": faithfulness,
    "answer_relevancy": AnswerRelevancy(strictness=1),
    "context_precision": context_precision,
    "context_recall": context_recall,
    "answer_correctness": answer_correctness,
}

results_summary = []

for record in raw_dataset:
    record_id = record["id"]
    contexts = record["retrieved_contexts"]
    context_len_chars = sum(len(c) for c in contexts)
    answer_len_chars = len(record["response"])

    ragas_records = [{
        "question": record["user_input"],
        "answer": record["response"],
        "contexts": contexts,
        "ground_truth": record["reference"],
    }]
    dataset = Dataset.from_list(ragas_records)

    for metric_name, metric in metrics.items():
        try:
            evaluate(
                dataset=dataset,
                metrics=[metric],
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
                run_config=RunConfig(max_workers=1, timeout=180, max_retries=1),
                raise_exceptions=True,
            )
            status = "OK"
            error = ""
        except Exception as e:  # noqa: BLE001
            status = "FAILED"
            error = str(e)[:200]

        results_summary.append({
            "record_id": record_id,
            "metric": metric_name,
            "status": status,
            "error": error,
            "context_len_chars": context_len_chars,
            "answer_len_chars": answer_len_chars,
        })
        print(f"{record_id:10} | {metric_name:20} | {status:6} | ctx={context_len_chars:6} ans={answer_len_chars:4} | {error}")

print("\n=== Summary of failures ===")
failures = [r for r in results_summary if r["status"] == "FAILED"]
if not failures:
    print("No failures found when run record-by-record. The issue may be "
          "specific to running the full batch (e.g. rate limiting, "
          "concurrency, or context window pressure across parallel jobs).")
else:
    for f in failures:
        print(f"{f['record_id']} / {f['metric']}: {f['error']}")
    print(f"\n{len(failures)} failing (record, metric) pairs out of {len(results_summary)} total.")