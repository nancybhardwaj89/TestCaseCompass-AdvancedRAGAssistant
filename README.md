# TestCase Compass — Advanced RAG Assistant

TestCase Compass is an Advanced RAG proof of concept that helps QA teams search and retrieve relevant information from a repository of 5,000 test cases using natural-language questions.

> Example query: **Show high-priority test cases for Patient Registration**

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

- **HyDE**  
  Generates an answer-like search query to improve retrieval for short or vague questions.

- **Semantic Search**  
  Uses Mistral embeddings and ChromaDB to find relevant test-case chunks by meaning.

- **NVIDIA Reranking**  
  Reorders search results and keeps the most relevant matches.

- **Parent Document Retrieval**  
  Retrieves small child chunks for accurate search, then uses `parent_id` to fetch the complete original test case.

- **Contextual Compression**  
  Removes irrelevant details before sending context to the final LLM.

- **Grounded Answer Generation**  
  Generates answers only from retrieved test-case context to reduce hallucinations.

## Architecture

### Offline Indexing

The 5,000 test cases are stored in two forms:

| Data Type | Purpose |
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
- Custom Langflow Components:
  - Parent Document Builder
  - Child Chunk Builder
  - Parent Document Resolver

## $0 Proof of Concept

This project was built as a **$0 proof of concept** using:

- Langflow Desktop
- Local ChromaDB
- Mistral embeddings
- Groq LLM
- NVIDIA free reranking endpoint

> Free-tier availability, quotas, and model access can change. This project is designed for learning, experimentation, and demonstration purposes.

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

It is about improving what reaches the LLM.

**Better retrieval → better grounded answers.**

## Author

**Nancy Bhardwaj**  
QA | AI Testing | RAG | Test Automation
