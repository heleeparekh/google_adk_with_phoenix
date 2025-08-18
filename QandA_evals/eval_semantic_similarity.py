# eval_semantic_similarity_embed.py
import os, re, json, ast, math
import pandas as pd
import phoenix as px
from dotenv import load_dotenv
from phoenix.trace import SpanEvaluations
from openai import OpenAI

load_dotenv()
PROJECT = os.getenv("PHOENIX_PROJECT", "adk-q-and-a")
CSV_PATH = os.getenv("RAG_QA_CSV", "rag_sample_qas_from_kis.csv")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")  # or text-embedding-3-large

# ---------- helpers ----------
def _safe_loads(cell):
    if isinstance(cell, dict):
        return cell
    if cell is None:
        return None
    s = str(cell)
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return None

def _detect_out_col(df: pd.DataFrame) -> str:
    for c in ["attributes.llm.output_messages", "attributes.llm.output_value"]:
        if c in df.columns:
            return c
    raise RuntimeError("No LLM output column found on call_llm spans.")

def _coerce_text(series: pd.Series) -> pd.Series:
    out = []
    for v in series:
        d = _safe_loads(v)
        if isinstance(d, list) and d and isinstance(d[0], dict):
            txt = d[0].get("content") or d[0].get("text") or str(v)
            out.append(str(txt))
        elif isinstance(d, dict):
            txt = d.get("content") or d.get("text") or str(v)
            out.append(str(txt))
        else:
            out.append(str(v))
    return pd.Series(out, index=series.index)

def _trace_to_row_map(spans_all: pd.DataFrame) -> pd.DataFrame:
    var_cols = [c for c in spans_all.columns if "variables" in c.lower() or "prompt" in c.lower()]
    rows = []
    for c in var_cols:
        s = spans_all[["context.trace_id", c]].dropna()
        for tid, cell in zip(s["context.trace_id"], s[c]):
            d = _safe_loads(cell)
            if isinstance(d, dict) and d.get("row_id_for_gt") is not None:
                try:
                    rows.append((tid, int(d["row_id_for_gt"])))
                except Exception:
                    pass
    return pd.DataFrame(rows, columns=["context.trace_id", "row_id"]).drop_duplicates()

def _normalize(s: str) -> str:
    return " ".join((s or "").strip().split())

def _sentences(s: str):
    s = _normalize(s)
    if not s:
        return []
    parts = re.split(r"(?<=[.!?])\s+", s)
    parts = [p.strip() for p in parts if p.strip()]
    return parts or [s]

def _batch(xs, n):
    buf = []
    for x in xs:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf

def _embed_texts(client: OpenAI, texts, model: str, batch_size: int = 96):
    vecs = []
    for chunk in _batch(texts, batch_size):
        resp = client.embeddings.create(model=model, input=chunk)
        vecs.extend([d.embedding for d in resp.data])
    return vecs

def _cosine(u, v):
    du = math.sqrt(sum(x*x for x in u))
    dv = math.sqrt(sum(y*y for y in v))
    if du == 0 or dv == 0:
        return 0.0
    return sum(x*y for x, y in zip(u, v)) / (du * dv)

# ---------- main ----------
def main():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = px.Client()
    spans_all = client.get_spans_dataframe(project_name=PROJECT)
    spans_llm = spans_all[spans_all["name"] == "call_llm"].copy()
    if spans_llm.empty:
        print("No call_llm spans found. Run your agent first.")
        return

    out_col = _detect_out_col(spans_llm)

    # Join ground truth via row_id_for_gt from any span in the same trace
    trace_to_row = _trace_to_row_map(spans_all)
    if trace_to_row.empty:
        raise RuntimeError("row_id_for_gt not found in trace variables. Ensure using_prompt_template logs it.")

    spans_llm = spans_llm.merge(trace_to_row, on="context.trace_id", how="left")

    df_csv = pd.read_csv(CSV_PATH).reset_index().rename(columns={"index": "row_id"})
    spans_llm = spans_llm.merge(df_csv[["row_id", "sample_ground_truth"]], on="row_id", how="left")

    df = spans_llm.dropna(subset=["sample_ground_truth"]).set_index("context.span_id")[
        [out_col, "sample_ground_truth"]
    ].rename(columns={out_col: "generated", "sample_ground_truth": "reference"})

    df["generated"] = _coerce_text(df["generated"]).map(_normalize)
    df["reference"] = df["reference"].astype(str).map(_normalize)
    df = df[(df["generated"] != "") & (df["reference"] != "")]
    if df.empty:
        print("No non-empty answer/reference pairs.")
        return

    # Sentence-level embeddings for coverage-style similarity
    oa = OpenAI(api_key=OPENAI_API_KEY)

    # Build sentence inventories and remember index ranges per row
    rows = []
    all_sent_texts = []
    for i, (span_id, row) in enumerate(df.iterrows()):
        ref_sents = _sentences(row["reference"])[:30]
        gen_sents = _sentences(row["generated"])[:30]
        start_ref = len(all_sent_texts)
        all_sent_texts.extend(ref_sents)
        start_gen = len(all_sent_texts)
        all_sent_texts.extend(gen_sents)
        rows.append((span_id, start_ref, start_gen, len(ref_sents), len(gen_sents), ref_sents, gen_sents))

    # Embed all sentences in one pass
    vecs = _embed_texts(oa, all_sent_texts, EMBED_MODEL)
    sent_idx = 0

    scores = {}
    explanations = {}

    # Threshold for marking a ground-truth sentence as uncovered in explanation
    uncovered_thresh = 0.60

    for span_id, start_ref, start_gen, n_ref, n_gen, ref_sents, gen_sents in rows:
        ref_vecs = vecs[start_ref : start_ref + n_ref]
        gen_vecs = vecs[start_gen : start_gen + n_gen]

        # For each ground-truth sentence, find its best match in the answer
        best_scores = []
        best_pairs = []
        for i, rv in enumerate(ref_vecs):
            best = -1.0
            best_j = -1
            for j, gv in enumerate(gen_vecs):
                c = _cosine(rv, gv)
                # scale cosine [-1,1] to [0,1]
                sim01 = (c + 1.0) / 2.0
                if sim01 > best:
                    best = sim01
                    best_j = j
            best_scores.append(best if best >= 0 else 0.0)
            pair = (ref_sents[i], gen_sents[best_j] if best_j >= 0 else "")
            best_pairs.append((pair, best if best >= 0 else 0.0))

        # Document score is mean best similarity over ground-truth sentences
        doc_score = sum(best_scores) / len(best_scores) if best_scores else 0.0
        scores[span_id] = doc_score

        # Build human-readable explanation without bullets
        top_pairs = sorted(best_pairs, key=lambda t: t[1], reverse=True)[:3]
        missing = [(s, sc) for (s, _), sc in best_pairs if sc < uncovered_thresh][:3]

        parts = []
        if top_pairs:
            parts.append("Top matches:")
            for k, ((ref_s, gen_s), sc) in enumerate(top_pairs, 1):
                parts.append(f"{k}. GT: {ref_s} | Ans: {gen_s} | Sim={sc:.2f}")
        if missing:
            parts.append("Least covered ground-truth sentences:")
            for k, (ref_s, sc) in enumerate(missing, 1):
                parts.append(f"{k}. {ref_s} | BestSim={sc:.2f}")
        if not parts:
            parts.append("All ground-truth sentences were well covered.")

        explanations[span_id] = "\n".join(parts)

    sim_df = pd.DataFrame({
        "context.span_id": list(scores.keys()),
        "score": [scores[k] for k in scores.keys()],
        "explanation": [explanations.get(k, "") for k in scores.keys()],
    }).set_index("context.span_id")
    client.log_evaluations(
        SpanEvaluations(eval_name="Response Ground Truth Semantic Similarity", dataframe=sim_df)
    )
    print(f"Logged {len(sim_df)} rows for 'Response Ground Truth Semantic Similarity' to project: {PROJECT}")
    print(sim_df.head())

if __name__ == "__main__":
    main()
