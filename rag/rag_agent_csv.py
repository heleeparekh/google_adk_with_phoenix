# rag_agent_csv.py — ADK RAG agent over CSV index (dict-safe pickle)

import os
import asyncio
import pickle
import re
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.tools.function_tool import FunctionTool
from google.genai import types

from phoenix.otel import register
from openinference.instrumentation import using_prompt_template

PROJECT_NAME = "adk-rag-csv"
INDEX_PATH = Path("corpus_index.pkl")

register(project_name=PROJECT_NAME, auto_instrument=True)

def _tok(s: str):
    return re.findall(r"\w+", s.lower())

def build_retriever_from_index():
    if not INDEX_PATH.exists():
        raise SystemExit(f"Missing {INDEX_PATH}. Run build_corpus_from_csv.py first.")
    with open(INDEX_PATH, "rb") as f:
        obj = pickle.load(f)
    chunks: List[Dict] = obj["chunks"]   # list of dicts
    bm25 = obj["bm25"]

    def retrieve(query: str, k: int = 4):
        # scores over the corpus tokens fit to bm25
        scores = bm25.get_scores(_tok(query))
        top = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        hits = []
        for idx, sc in top:
            c = chunks[idx]  # dict with text/id/row_id/source
            hits.append({
                "id": c["id"],
                "row_id": c["row_id"],
                "score": float(sc),
                "text": c["text"],
                "source": c["source"],
            })
        return hits
    return retrieve

_retrieve = build_retriever_from_index()

def retrieve_docs(query: str, k: int = 4) -> str:
    hits = _retrieve(query, k=k)
    return "\n\n".join(f"[{h['id']}] {h['text']}" for h in hits)

retrieve_tool = FunctionTool(retrieve_docs)

MODEL = os.getenv("MODEL", "gemini-2.0-flash-001")

rag_instruction = """
You are a RAG agent. Always use the provided retrieved context to answer.
If the context does not contain the answer, say you don't know.
Keep answers concise and cite chunk ids if useful (e.g., [row::12__chunk::0]).
"""

rag_agent = Agent(
    model=MODEL,
    name="adk_rag_agent",
    description="RAG over CSV Q&A corpus",
    instruction=rag_instruction,
    tools=[retrieve_tool],
    generate_content_config=types.GenerateContentConfig(temperature=0.0),
)

async def run_rag(query: str, query_id: str = ""):
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()
    app_name = "adk_rag_csv_demo"

    await session_service.create_session(app_name=app_name, user_id="u", session_id="s")
    runner = Runner(app_name=app_name, agent=rag_agent,
                    artifact_service=artifact_service, session_service=session_service)

    # 1) Retrieve
    context_text = retrieve_docs(query, k=4)

    # 2) Build the ACTUAL prompt that includes the context
    message_text = (
        f"Query {query_id}: {query}\n\n"
        f"Retrieved Context:\n{context_text}\n\n"
        f"Answer:"
    )

    # 3) Record a prompt template for Phoenix AND send the same text to the model
    with using_prompt_template(
        template="Query {query_id}: {query}\n\nRetrieved Context:\n{context}\n\nAnswer:",
        variables={"query": query, "query_id": query_id, "context": context_text},
        version="v1",
    ):
        content = types.Content(role="user", parts=[types.Part(text=message_text)])
        events = runner.run_async(user_id="u", session_id="s", new_message=content)

        final = None
        async for ev in events:
            if ev.is_final_response() and ev.content and ev.content.parts:
                final = ev.content.parts[0].text
        return context_text, final

if __name__ == "__main__":
    ctx, ans = asyncio.run(run_rag(
        "Explain Attention Mechanism briefly.",
        query_id="demo1"
    ))
    print("Context:\n", ctx)
    print("\nAnswer:\n", ans)
