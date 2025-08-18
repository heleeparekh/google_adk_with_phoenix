import os
import re
import uuid
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from dotenv import load_dotenv; load_dotenv()

from rank_bm25 import BM25Okapi

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
from google.genai import types

from phoenix.otel import register
from openinference.instrumentation import using_prompt_template

PROJECT = os.getenv("PHOENIX_PROJECT", "adk-q-and-a")
MODEL = os.getenv("MODEL", "gemini-2.0-flash-001")
APP_NAME = os.getenv("APP_NAME", "adk_rag_minimal")

CSV_PATH = Path("rag_sample_qas_from_kis.csv")

MAX_CHARS_PER_CHUNK = 1200
MAX_OVERLAP_CHARS = 150
TOP_K = 4
TEMPERATURE = 0.0

register(project_name=PROJECT, auto_instrument=True)

def _tok(s: str) -> List[str]:
    return re.findall(r"\w+", (s or "").lower())

def _chunk_text(s: str, max_len: int, overlap: int) -> List[str]:
    s = s or ""
    if len(s) <= max_len:
        return [s]
    out, i = [], 0
    step = max_len - overlap
    while i < len(s):
        out.append(s[i:i+max_len])
        if i + max_len >= len(s):
            break
        i += step
    return out

class Retriever:
    def __init__(self, df: pd.DataFrame):
        for col in ["ki_topic", "ki_text", "sample_question", "sample_ground_truth"]:
            if col not in df.columns:
                raise ValueError(f"CSV missing column: {col}")

        self.rows = df.reset_index(drop=True)
        self.chunks: List[Dict[str, Any]] = []
        for row_id, row in self.rows.iterrows():
            topic = str(row["ki_topic"])
            article = str(row["ki_text"])
            pieces = _chunk_text(article, MAX_CHARS_PER_CHUNK, MAX_OVERLAP_CHARS)
            for i, text in enumerate(pieces):
                self.chunks.append({
                    "chunk_id": f"row::{row_id}__chunk::{i}",
                    "row_id": row_id,
                    "source": topic,
                    "text": text,
                })
        self._tokenized = [_tok(c["text"]) for c in self.chunks]
        self._bm25 = BM25Okapi(self._tokenized)

    def retrieve(self, query: str, k: int = TOP_K) -> Dict[str, Any]:
        scores = self._bm25.get_scores(_tok(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results = []
        for rank, idx in enumerate(order, start=1):
            c = self.chunks[idx]
            results.append({
                "chunk_id": c["chunk_id"],
                "row_id": c["row_id"],
                "rank": rank,
                "score": float(scores[idx]),
                "source": c["source"],
                "text": c["text"],
            })
        # No bracketed IDs in the fused context sent to the model
        context_text = "\n\n".join(r["text"] for r in results)
        return {"top_k": k, "results": results, "context_text": context_text}

_df = pd.read_csv(CSV_PATH)
_RETR = Retriever(_df)

INSTRUCTION = (
    "You are a strict RAG agent. Use only the provided context. "
    "If the context does not contain the answer, reply that you don't know. "
    "Do not include any internal IDs or citations in your answer. "
    "Be concise."
)

agent = Agent(
    model=MODEL,
    name="adk_rag_minimal_agent",
    description="Minimal RAG over CSV corpus",
    instruction=INSTRUCTION,
    tools=[],
    generate_content_config=types.GenerateContentConfig(temperature=TEMPERATURE),
)

_session = InMemorySessionService()
_artifacts = InMemoryArtifactService()
_runner: Optional[Runner] = None

async def _ensure_runner() -> Runner:
    global _runner
    if _runner is None:
        _runner = Runner(app_name=APP_NAME, agent=agent,
                         artifact_service=_artifacts, session_service=_session)
    return _runner

async def run_rag(
    query: str,
    k: int = TOP_K,
    row_id_for_gt: Optional[int] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    runner = await _ensure_runner()

    sid = session_id or f"s-{uuid.uuid4().hex[:12]}"
    await _session.create_session(app_name=APP_NAME, user_id="u", session_id=sid)

    retr = _RETR.retrieve(query, k=k)
    context_text = retr["context_text"]

    template = (
        "Query: {query}\n\n"
        "Context:\n{context}\n\n"
        "Answer:"
    )
    variables = {
        "query": query,
        "context": context_text,
        "row_id_for_gt": row_id_for_gt,
        "retrieval_chunk_ids": [r["chunk_id"] for r in retr["results"]],
        "retrieval_row_ids": [r["row_id"] for r in retr["results"]],
        "retrieval_scores": [r["score"] for r in retr["results"]],
        "top_k": k,
    }
    prompt_text = template.format(query=query, context=context_text)

    with using_prompt_template(template=template, variables=variables, version="v1"):
        content = types.Content(role="user", parts=[types.Part(text=prompt_text)])
        events = runner.run_async(user_id="u", session_id=sid, new_message=content)

        final = None
        async for ev in events:
            if ev.is_final_response() and ev.content and ev.content.parts:
                final = ev.content.parts[0].text

    record = {
        "query": query,
        "context": context_text,
        "answer": final,
        "row_id_for_gt": row_id_for_gt,
        "session_id": sid,
    }
    if row_id_for_gt is not None and 0 <= row_id_for_gt < len(_df):
        record["ground_truth"] = str(_df.iloc[row_id_for_gt]["sample_ground_truth"])
    return record

if __name__ == "__main__":
    batch_queries = [
        "How do I set up my company email on my mobile device?",
        "I forgot my PIN, how can I reset it?",
        "How do I set up VPN access on my laptop so I can work from home and access company resources?",
        "My Microsoft Word keeps freezing every time I try to open a document, I've tried closing and reopening it but the issue persists, what can I do to fix it?",
        "How do I set up a conference call on Cisco Webex with both video and audio for a meeting with multiple participants?",
        "How do I back up my important work files to prevent data loss?",
        "My company-issued tablet is freezing frequently and I'm unable to access some of my apps, what can I do to fix the issue?",
    ]

    async def run_batch():
        for i, q in enumerate(batch_queries):
            sid = f"s-{uuid.uuid4().hex[:12]}"
            out = await run_rag(q, k=4, row_id_for_gt=i, session_id=sid)
            print(f"\n=== Query {i+1} | session_id={sid} ===")
            print("Query:", out["query"])
            print("\nContext:\n", out["context"][:1200], "...\n")
            print("Answer:", out["answer"])
            if "ground_truth" in out:
                print("\nGround Truth:", out["ground_truth"])

    asyncio.run(run_batch())
