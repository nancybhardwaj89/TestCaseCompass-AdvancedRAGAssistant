import json
import os
from pathlib import Path

import pandas as pd
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
RESULTS_PATH = Path("results/ragas_results.csv")


def main():
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is missing in the .env file.")

    if not os.getenv("MISTRAL_API_KEY"):
        raise ValueError("MISTRAL_API_KEY is missing in the .env file.")

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        raw_dataset = json.load(file)

    # RAGAS legacy evaluation format.
    ragas_records = []

    for record in raw_dataset:
        ragas_records.append(
            {
                "id": record["id"],
                "question": record["user_input"],
                "answer": record["response"],
                "contexts": record["retrieved_contexts"],
                "ground_truth": record["reference"],
            }
        )

    dataset = Dataset.from_list(ragas_records)

    evaluator_llm = LangchainLLMWrapper(
        ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
             max_tokens=4096,
        )
    )

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        MistralAIEmbeddings(
            model="mistral-embed",
        )
    )

    # Groq's API rejects n > 1 ("'n' : number must be at most 1").
    # RAGAS's default answer_relevancy asks the LLM to generate 3
    # candidate questions per answer (n=3), which fails silently
    # against Groq and gets scored as 0.0 for every row. Setting
    # strictness=1 makes RAGAS request only 1 generated question,
    # which Groq accepts. Trade-off: the relevancy score is now based
    # on a single generated question instead of an average of 3, so
    # it will be slightly noisier per-row than the RAGAS default.
    answer_relevancy_metric = AnswerRelevancy(strictness=1)

    print("Starting RAGAS evaluation...")

    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy_metric,
            context_precision,
            context_recall,
            answer_correctness,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=RunConfig(
            max_workers=1,
            timeout=180,
            max_retries=3,
        ),
    )

    results_df = results.to_pandas()

    Path("results").mkdir(exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)

    print("\nRAGAS Evaluation Results\n")
    print(results_df.to_string(index=False))

    metric_columns = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness",
    ]

    print("\nAverage Scores\n")
    print(results_df[metric_columns].mean().round(3))

    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()