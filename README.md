# TestCase Compass — Advanced RAG Assistant

TestCase Compass is an Advanced RAG proof of concept that helps QA teams search and retrieve relevant information from a repository of **5,000 synthetic test cases** using natural-language questions.

> Example query: **Show high-priority functional test cases for Patient Registration**

## Overview

Basic RAG retrieves similar chunks and sends them to an LLM.

TestCase Compass goes further by improving the quality of retrieved context before generating the final answer.

```text
User Question
→ HyDE
→ Semantic Search
→ NVIDIA Rerank
→ Parent Document Retrieval
→ Contextual Compression
→ Grounded Answer
```

## Advanced RAG Features

- **HyDE** — Generates an answer-like search query to improve retrieval for short or vague questions.
- **Semantic Search** — Uses Mistral embeddings and ChromaDB to find relevant test-case chunks by meaning.
- **NVIDIA Reranking** — Reorders retrieved results and keeps the most relevant matches.
- **Parent Document Retrieval** — Retrieves small child chunks for accurate search, then uses `parent_id` to fetch the complete source test case.
- **Contextual Compression** — Removes irrelevant details before sending context to the final LLM.
- **Grounded Answer Generation** — Generates answers only from retrieved test-case context to reduce hallucinations.

## Architecture

### Offline Indexing

The 5,000 test cases are stored in two forms:

| Data type | Purpose |
| --- | --- |
| Parent Documents | Stores one complete test case, including steps and expected results. |
| Child Chunks | Stores smaller searchable chunks with `parent_id` metadata. |

Each child chunk retains its `parent_id`, allowing the pipeline to retrieve the full source test case after reranking.

### Live Retrieval

```text
User Question
→ HyDE Query Generation
→ Semantic Search on Child Chunks
→ NVIDIA Rerank
→ Parent Document Resolver
→ Contextual Compression
→ Final Grounded Answer
```

## Technology Stack

- Langflow Desktop
- ChromaDB
- Mistral `mistral-embed`
- NVIDIA `rerank-qa-mistral-4b`
- Groq `llama-3.1-8b-instant`
- RAGAS for RAG evaluation
- React + Vite dashboard for visualizing evaluation results
- Custom Langflow components:
  - Parent Document Builder
  - Child Chunk Builder
  - Parent Document Resolver

## RAGAS Evaluation

Building a RAG pipeline is not enough—the pipeline also needs to be measured for retrieval and answer quality.

This project uses **RAGAS** with a curated evaluation set containing natural-language, filter-based, and exact-ID test-case queries.

For each evaluation case, the dataset captures:

```text
User question
→ Actual retrieved parent-document context
→ Actual generated answer
→ Source-based reference answer
→ RAGAS scores
```

### Metrics Evaluated

| Metric | What it measures |
| --- | --- |
| Faithfulness | Whether the answer is supported by retrieved context. |
| Answer Relevancy | Whether the answer directly addresses the user's question. |
| Context Precision | Whether retrieved documents are relevant to the question. |
| Context Recall | Whether the required source information was retrieved. |
| Answer Correctness | How closely the generated answer matches the source-based reference answer. |

### Baseline Evaluation Results

Latest baseline over 10 curated evaluation queries:

| Metric | Score |
| --- | ---: |
| Faithfulness | 0.775 |
| Answer Relevancy | 0.831 |
| Context Precision | 0.900 |
| Context Recall | 0.867 |
| Answer Correctness | 0.716 |

The evaluation found that natural-language retrieval performed well, while exact test-case ID searches need an exact-match or hybrid retrieval path in addition to semantic search.

### Run Evaluation

```bash
python run_ragas.py
```

RAGAS results are saved to:

```text
results/ragas_results.csv
```

### React Evaluation Dashboard

The lightweight React dashboard reads `ragas_results.csv` and displays:

- Overall metric score cards
- Per-case evaluation scores
- User question, generated answer, reference answer, and retrieved context
- Low-scoring cases for investigation

```bash
cd ragas-dashboard
npm install
npm run dev
```

Copy the latest results file into the dashboard before opening it:

```bash
copy ..\results\ragas_results.csv .\public\results\ragas_results.csv
```

## $0 Proof of Concept

This project was built as a **$0 learning-focused proof of concept** using:

- Langflow Desktop
- Local ChromaDB
- Mistral free-tier embeddings access
- Groq free-tier LLM access
- NVIDIA's free reranking endpoint for experimentation

> Free-tier availability, quotas, and model access can change. This repository is designed for learning, experimentation, and demonstration—not production use.

## Example Questions

- Show high-priority test cases for Patient Registration
- Find negative test cases for appointment scheduling
- Show security test cases for the Payment module
- Retrieve test cases related to telehealth
- Show validation test cases for user registration

## Setup

1. Install and launch Langflow Desktop.
2. Import the Langflow flow JSON.
3. Add your API keys in Langflow:
   - `MISTRAL_API_KEY`
   - `GROQ_API_KEY`
   - `NVIDIA_API_KEY`
4. Upload the test-case CSV file.
5. Run the parent-document and child-chunk ingestion nodes.
6. Open Langflow Playground and ask a question.

Use the same ChromaDB persist directory for all Chroma nodes:

```text
./chroma_db
```

## Key Learning

Advanced RAG is not only about selecting an LLM.

It is about improving retrieval **and** continuously measuring whether the retrieved context produces grounded, relevant, and correct answers.

**Better retrieval + evaluation → better grounded answers.**

## Author

**Nancy Bhardwaj**  
QA | AI Testing | RAG | Test Automation
