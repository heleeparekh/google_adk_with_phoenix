# eval_query_context_relevance_min.py
# Query Context Relevance & Supporting Evidence (embedding-based, compact)

import os, re, math, json, pandas as pd, phoenix as px
from dotenv import load_dotenv
from phoenix.trace import SpanEvaluations
from openai import OpenAI

load_dotenv()
PROJECT = os.getenv("PHOENIX_PROJECT", "adk-q-and-a")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")  # or 'text-embedding-3-large'

# Tunables
TOPK = 5          # number of most relevant context sentences to average/show
MAX_CTX = 200     # cap sentences per context to control cost

# Split context into sentences / list items
SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+\s*\d+[.)-]\s+|^\s*[-*•]\s+|\n\s*[-*•]\s+|;|[\r\n]+")
def sents(s: str):
    return [p.strip() for p in SPLIT_RE.split((s or "").strip()) if p.strip()]

def cosine(u, v):
    du = math.sqrt(sum(x*x for x in u)); dv = math.sqrt(sum(y*y for y in v))
    if du == 0 or dv == 0: return 0.0
    return sum(x*y for x,y in zip(u,v)) / (du*dv)

def sim01(u, v):  # [-1,1] -> [0,1]
    c = max(min(cosine(u, v), 1.0), -1.0)
    return (c + 1.0) / 2.0

def main():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    oa = OpenAI(api_key=OPENAI_API_KEY)

    client = px.Client()
    spans = client.get_spans_dataframe(project_name=PROJECT)
    spans = spans[spans["name"] == "call_llm"].copy()
    if spans.empty:
        print("No call_llm spans found."); return

    # Minimal assumption: prompt variables (with 'query' and 'context') are on this span
    var_col = next((c for c in spans.columns if "variables" in c.lower()), None)
    if not var_col:
        raise RuntimeError("No prompt variables column found. Ensure using_prompt_template logs {query, context}.")

    # Build (span_id, query, context) rows
    rows = []
    for _, r in spans.iterrows():
        v = r[var_col]
        if not isinstance(v, dict):
            try: v = json.loads(str(v))
            except Exception: v = {}
        rows.append((r["context.span_id"], str(v.get("query", "")), str(v.get("context", ""))))
    df = pd.DataFrame(rows, columns=["context.span_id", "query", "context"]).set_index("context.span_id")

    # Filter empties
    df["query"] = df["query"].str.strip()
    df["context"] = df["context"].str.strip()
    df = df[(df["query"] != "") & (df["context"] != "")]
    if df.empty:
        print("No non-empty (query, context) pairs."); return

    scores, explanations = {}, {}

    for span_id, row in df.iterrows():
        q = row["query"]
        C = sents(row["context"])[:MAX_CTX]
        if not C:
            scores[span_id] = 0.0
            explanations[span_id] = "Empty context."
            continue

        # Single batched embeddings call per span
        texts = [q] + C
        resp = oa.embeddings.create(model=EMBED_MODEL, input=texts)
        qv = resp.data[0].embedding
        Cvec = [resp.data[i].embedding for i in range(1, 1 + len(C))]

        # Query↔each context sentence similarity
        sims = [sim01(qv, cv) for cv in Cvec]
        # Top-K average as relevance score (robust to long contexts)
        K = min(TOPK, len(sims))
        top_idx = sorted(range(len(sims)), key=lambda j: sims[j], reverse=True)[:K]
        score = sum(sims[j] for j in top_idx) / K if K > 0 else 0.0
        scores[span_id] = score

        # Supporting evidence: show top-K context sentences with their sims
        lines = [f"Query–Context relevance: {score:.2f} (Top-{K} avg).", "Top supporting context sentences:"]
        for rank, j in enumerate(top_idx, 1):
            lines.append(f"{rank}. {C[j]} | Sim={sims[j]:.2f}")
        explanations[span_id] = "\n".join(lines)

    # Log a single Phoenix metric with both score + explanation
    eval_df = pd.DataFrame({"score": pd.Series(scores), "explanation": pd.Series(explanations)})
    eval_df.index.name = "context.span_id"
    client.log_evaluations(SpanEvaluations(
        eval_name="Query Context Relevance & Supporting Evidence",
        dataframe=eval_df
    ))
    print(f"Logged {len(eval_df)} rows to project: {PROJECT}")

if __name__ == "__main__":
    main()
