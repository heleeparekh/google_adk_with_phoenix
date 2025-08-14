import os, re, pickle, asyncio
from pathlib import Path

from dotenv import load_dotenv; load_dotenv()

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.tools.function_tool import FunctionTool
from google.genai import types

from phoenix.otel import register
from openinference.instrumentation import using_prompt_template

# ---- config ----
PROJECT = "adk-rag-csv"
INDEX_PATH = Path("corpus_index.pkl")
MODEL = os.getenv("MODEL", "gemini-2.0-flash-001")

register(project_name=PROJECT, auto_instrument=True)

# ---- tiny retriever over pickled index ----
def _tok(s: str): return re.findall(r"\w+", s.lower())

with open(INDEX_PATH, "rb") as f:
    obj = pickle.load(f)  # {"chunks": [ {id,row_id,text,source}, ...], "bm25": BM25Okapi}
_chunks, _bm25 = obj["chunks"], obj["bm25"]

def retrieve_docs(query: str, k: int = 4) -> str:
    scores = _bm25.get_scores(_tok(query))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    hits = [f"[{_chunks[i]['id']}] {_chunks[i]['text']}" for i in order]
    return "\n\n".join(hits)

retrieve_tool = FunctionTool(retrieve_docs)

# ---- agent ----
rag_instruction = (
    "You are a RAG agent. Use only the retrieved context. "
    "If context lacks the answer, say you don't know. Be concise; cite chunk ids like [row::12__chunk::0]."
)
agent = Agent(
    model=MODEL,
    name="adk_rag_agent",
    description="RAG over CSV Q&A corpus",
    instruction=rag_instruction,
    tools=[retrieve_tool],
    generate_content_config=types.GenerateContentConfig(temperature=0.0),
)

# ---- singletons for fast calls ----
_session = InMemorySessionService()
_artifacts = InMemoryArtifactService()
_runner: Runner | None = None
_APP = "adk_rag_csv_demo"

async def _ensure_runner():
    global _runner
    if _runner is None:
        await _session.create_session(app_name=_APP, user_id="u", session_id="s")
        _runner = Runner(app_name=_APP, agent=agent,
                         artifact_service=_artifacts, session_service=_session)
    return _runner

# ---- public API ----
async def run_rag(query: str, query_id: str = ""):
    runner = await _ensure_runner()
    context = retrieve_docs(query, k=4)

    # record vars for evals + send identical text to the model
    prompt_text = f"Query {query_id}: {query}\n\nRetrieved Context:\n{context}\n\nAnswer:"
    with using_prompt_template(
        template="Query {query_id}: {query}\n\nRetrieved Context:\n{context}\n\nAnswer:",
        variables={"query": query, "query_id": query_id, "context": context},
        version="v1",
    ):
        content = types.Content(role="user", parts=[types.Part(text=prompt_text)])
        events = runner.run_async(user_id="u", session_id="s", new_message=content)

        final = None
        async for ev in events:
            if ev.is_final_response() and ev.content and ev.content.parts:
                final = ev.content.parts[0].text
        return context, final

# quick manual run
if __name__ == "__main__":
    ctx, ans = asyncio.run(run_rag("Explain Attention Mechanism briefly.", query_id="demo1"))
    print("Context:\n", ctx)
    print("\nAnswer:\n", ans)
