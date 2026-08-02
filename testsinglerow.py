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
from ragas.metrics import answer_relevancy
from ragas.run_config import RunConfig

load_dotenv()

DATASET_PATH = Path("datasets/golden_dataset.json")
TEST_RECORD_ID = "RAG_001"  # pick a non-fallback row

with open(DATASET_PATH, "r", encoding="utf-8") as file:
    raw_dataset = json.load(file)

record = next(r for r in raw_dataset if r["id"] == TEST_RECORD_ID)

print(f"Testing record: {record['id']}")
print(f"Question: {record['user_input']}")
print(f"Answer (first 200 chars): {record['response'][:200]}")
print("---")

ragas_records = [{
    "question": record["user_input"],
    "answer": record["response"],
    "contexts": record["retrieved_contexts"],
    "ground_truth": record["reference"],
}]

dataset = Dataset.from_list(ragas_records)

evaluator_llm = LangchainLLMWrapper(
    ChatGroq(model="llama-3.1-8b-instant", temperature=0)
)
evaluator_embeddings = LangchainEmbeddingsWrapper(
    MistralAIEmbeddings(model="mistral-embed")
)

results = evaluate(
    dataset=dataset,
    metrics=[answer_relevancy],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
    run_config=RunConfig(max_workers=1, timeout=180, max_retries=3),
    raise_exceptions=True,  # <-- this is the key change: surfaces real errors
)

print(results.to_pandas())