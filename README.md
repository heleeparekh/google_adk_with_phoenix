# ADK Walkthrough: Google ADK Agents with Arize Phoenix Tracing

This project demonstrates how to create and manage different agents using the [Google Agent Development Kit (ADK)](https://github.com/google/agent-development-kit) and integrate them with [Arize Phoenix](https://github.com/Arize-ai/phoenix) for tracing and observability.

## Features

- Build and configure multiple agents using Google ADK
- Integrate agents with Arize Phoenix for end-to-end tracing
- Example workflows and tracing visualization

## Getting Started

### Prerequisites

- Python 3.8+
- [Google ADK](https://google.github.io/adk-docs/)
- [Arize Phoenix](https://arize.com/docs/phoenix)

### Installation

1. Clone this repository:
    ```sh
    git clone https://github.com/heleeparekh/google_adk_with_phoenix.git
    cd google_adk_with_phoenix
    ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

### Usage


1. **Start the Phoenix server** (required for tracing):
    ```sh
    phoenix serve
    ```
2. **Set required environment variables** for authentication and configuration.
3. **Run an agent script** of your choice to start an agent and enable tracing. For example:
    ```sh
    python agents/coding_agent.py
    ```
    or
    ```sh
    python agents/math_assistant.py
    ```
4. **View traces** in the Arize Phoenix dashboard.
5. **(Optional) Run RAG or evaluation scripts** as needed:
    ```sh
    python rag/rag_agent_csv.py
    python online_evals/eval_rag.py
    ```

## RAG (Retrieval-Augmented Generation) Workflow

The `rag/` folder demonstrates how to build and use a RAG pipeline over a CSV knowledge base, and how to evaluate its performance.

### RAG Scripts

- **build_corpus_from_csv.py**  
  Builds a searchable corpus from a CSV file (`rag_sample_qas_from_kis.csv`).  
  This script chunks the knowledge base text and creates a BM25 index for efficient retrieval.  
  **Run this first** to generate `corpus_index.pkl`:
  ```sh
  python rag/build_corpus_from_csv.py
  ```

- **rag_agent_csv.py**  
  Runs a single RAG query using the indexed corpus.  
  It retrieves relevant context chunks for a user query and generates an answer using an agent.  
  Example usage:
  ```sh
  python rag/rag_agent_csv.py
  ```

- **run_batch.py**  
  Runs a batch of queries (sampled from the CSV) through the RAG agent.  
  Useful for testing and generating multiple traces for evaluation.
  ```sh
  python rag/run_batch.py
  ```

### RAG Data

- **rag_sample_qas_from_kis.csv**  
  The sample CSV file containing knowledge items, questions, and ground truth answers.
- **corpus_index.pkl**  
  The serialized BM25 index and chunked corpus, created by `build_corpus_from_csv.py`.

---

## Evaluation Workflow

The `online_evals/` folder contains scripts to evaluate the quality of RAG agent outputs using LLM-based metrics.

- **eval_rag.py**  
  Evaluates the RAG agent’s traces stored in Arize Phoenix.  
  It pulls LLM spans from the Phoenix project and runs two main evaluations:
  - **Context Relevancy:** Checks if the retrieved context is relevant to the query.
  - **Answer Faithfulness:** Checks if the generated answer is grounded in the retrieved context (not hallucinated).
  
  Results are logged back to Phoenix for visualization and analysis.
  ```sh
  python online_evals/eval_rag.py
  ```

## Typical RAG + Evaluation Workflow

1. **Build the corpus index:**  
   `python rag/build_corpus_from_csv.py`
2. **Run RAG agent(s) to generate traces:**  
   - Single query: `python rag/rag_agent_csv.py`
   - Batch queries: `python rag/run_batch.py`
3. **Evaluate the traces in Phoenix:**  
   `python online_evals/eval_rag.py`
4. **View results and evaluations in the Arize Phoenix dashboard.**

