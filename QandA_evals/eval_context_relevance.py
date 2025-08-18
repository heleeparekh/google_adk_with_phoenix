# eval_context_relevance_min.py
# Context Response Relevance & Supporting Evidence (compact version)

import os, re, math, json, pandas as pd, phoenix as px
from dotenv import load_dotenv
from phoenix.trace import SpanEvaluations
from openai import OpenAI

load_dotenv()
PROJECT = os.getenv("PHOENIX_PROJECT", "adk-q-and-a")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")  # or ...-large

# Tunables
Q_TAU = 0.50        # below this query-relevance, an answer sentence gets weight 0
TOPK = 3            # how many evidence pairs to show
MAX_ANS = 30        # cap answer sentences per span
MAX_CTX = 120       # cap context sentences per span

# Simple sentence splitter: ends, list numbers, bullets, semicolons, newlines
SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+\s*\d+[.)-]\s+|^\s*[-*•]\s+|\n\s*[-*•]\s+|;|[\r\n]+")

def sents(s: str):
    return [p.strip() for p in SPLIT_RE.split((s or "").strip()) if p.strip()]

def cosine(u, v):
    du = math.sqrt(sum(x*x for x in u)); dv = math.sqrt(sum(y*y for y in v))
    if du == 0 or dv == 0: return 0.0
    return sum(x*y for x, y in zip(u, v)) / (du * dv)

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

    # Pick LLM output column and variables (assumed on same span for minimal code)
    out_col = "attributes.llm.output_value" if "attributes.llm.output_value" in spans.columns else "attributes.llm.output_messages"
    var_col = next((c for c in spans.columns if "variables" in c.lower()), None)
    if out_col not in spans.columns or not var_col:
        raise RuntimeError("Expected LLM output and prompt variables on call_llm spans.")

    # Build minimal eval frame: span_id, answer (as str), context, query
    rows = []
    for _, r in spans.iterrows():
        vars_cell = r[var_col]
        if not isinstance(vars_cell, dict):
            try:
                vars_cell = json.loads(str(vars_cell))
            except Exception:
                vars_cell = {}
        rows.append((
            r["context.span_id"],
            str(r[out_col]),
            str(vars_cell.get("context", "")),
            str(vars_cell.get("query", "")),
        ))
    df = pd.DataFrame(rows, columns=["context.span_id", "answer", "context", "query"]).set_index("context.span_id")
    df = df[(df["answer"].str.strip() != "") & (df["context"].str.strip() != "")]
    if df.empty:
        print("No non-empty (answer, context) pairs."); return

    scores, explanations = {}, {}
    for span_id, row in df.iterrows():
        A = sents(row["answer"])[:MAX_ANS]
        C = sents(row["context"])[:MAX_CTX]
        q = (row["query"] or "").strip()
        if not A or not C:
            scores[span_id] = 0.0
            explanations[span_id] = "Empty answer or context."
            continue

        # One batched embeddings call per span
        texts = [q] + A + C
        resp = oa.embeddings.create(model=EMBED_MODEL, input=texts)
        qv = resp.data[0].embedding
        Avec = [resp.data[i].embedding for i in range(1, 1 + len(A))]
        Cvec = [resp.data[i].embedding for i in range(1 + len(A), 1 + len(A) + len(C))]

        # For each answer sentence: weight by query relevance; score by best context support
        contrib = []
        for ai, av in enumerate(Avec):
            qrel = sim01(av, qv)                               # answer sentence ≃ query
            w = 0.0 if qrel < Q_TAU else (qrel - Q_TAU)/(1-Q_TAU)
            best, bj = 0.0, -1
            for cj, cv in enumerate(Cvec):
                s = sim01(av, cv)                              # answer sentence ≃ context sentence
                if s > best: best, bj = s, cj
            contrib.append((w, best, ai, bj, qrel))

        denom = sum(w for w, *_ in contrib)
        score = (sum(w*s for w, s, *_ in contrib) / denom) if denom > 1e-9 else 0.0
        scores[span_id] = score

        # Evidence: top contributors by w*s
        contrib.sort(key=lambda t: t[0]*t[1], reverse=True)
        lines = [f"Query-aware relevance: {score:.2f} (Qτ={Q_TAU}).", "Top supporting evidence:"]
        for k, (w, s, ai, bj, qrel) in enumerate(contrib[:TOPK], 1):
            ans_sent = A[ai]
            ctx_sent = C[bj] if 0 <= bj < len(C) else ""
            lines.append(f"{k}. Ans: {ans_sent} | Ctx: {ctx_sent} | Ans≃Ctx={s:.2f}, weight={w:.2f}, Ans≃Query={qrel:.2f}")
        explanations[span_id] = "\n".join(lines)

    # Log a single metric (score + explanation)
    eval_df = pd.DataFrame({"score": pd.Series(scores), "explanation": pd.Series(explanations)})
    eval_df.index.name = "context.span_id"
    client.log_evaluations(SpanEvaluations(
        eval_name="Context Response Relevance & Supporting Evidence",
        dataframe=eval_df
    ))
    print(f"Logged {len(eval_df)} rows to project: {PROJECT}")

if __name__ == "__main__":
    main()
