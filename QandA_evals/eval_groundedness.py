import os
import re
import json
import ast
import pandas as pd
import phoenix as px
from dotenv import load_dotenv
from phoenix.trace import SpanEvaluations
from phoenix.evals import OpenAIModel, llm_generate

load_dotenv()
PROJECT = os.getenv("PHOENIX_PROJECT", "adk-q-and-a")
CSV_PATH = os.getenv("RAG_QA_CSV", "rag_sample_qas_from_kis.csv")
JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

TEMPLATE = """
Return only a JSON object with keys "score" and "explanation". No prose, no code fences.

You are scoring how well the Generated Answer COVERS the Ground Truth.

1) Split the Ground Truth into minimal factual statements.
2) For each ground-truth statement, check if the Generated Answer clearly contains the same information
   (exactly or as an unambiguous paraphrase). If missing, contradicted, or too vague, mark it as not covered.
3) Score = (# ground-truth statements covered) / (total ground-truth statements). Return a float in [0,1].
4) Provide a brief explanation listing which ground-truth statements were covered and which were missing.

Ground Truth:
{reference}

Generated Answer:
{generated}
"""

def _safe_loads(cell):
    try:
        return cell if isinstance(cell, dict) else json.loads(cell)
    except Exception:
        try:
            return ast.literal_eval(cell)
        except Exception:
            return None

def _candidate_var_cols(df: pd.DataFrame):
    out = []
    for c in df.columns:
        cl = c.lower()
        if "variables" in cl or "prompt" in cl:
            out.append(c)
    return out

def _build_trace_to_row_map(spans_all: pd.DataFrame) -> pd.DataFrame:
    var_cols = _candidate_var_cols(spans_all)
    rows = []
    for c in var_cols:
        s = spans_all[["context.trace_id", c]].dropna()
        for trace_id, cell in zip(s["context.trace_id"], s[c]):
            d = _safe_loads(cell)
            if isinstance(d, dict) and "row_id_for_gt" in d:
                rid = d.get("row_id_for_gt")
                try:
                    rid = int(rid) if rid is not None else None
                except Exception:
                    rid = None
                if rid is not None:
                    rows.append((trace_id, rid))
    if not rows:
        return pd.DataFrame(columns=["context.trace_id", "row_id"]).astype({"context.trace_id": str, "row_id": "Int64"})
    return pd.DataFrame(rows, columns=["context.trace_id", "row_id"]).drop_duplicates()

def _row_from_answer(txt):
    m = re.search(r"row::(\d+)__chunk::\d+", str(txt))
    return int(m.group(1)) if m else None

def _safe_json_parser(out, _row):
    s = (out or "").strip()
    if not s:
        return {"score": None, "explanation": "empty judge output"}
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        m = re.search(r"(?i)score[^0-9]*([01](?:\.\d+)?)", s)
        if m:
            try:
                return {"score": float(m.group(1)), "explanation": s[:500]}
            except Exception:
                return {"score": None, "explanation": f"non-JSON judge output: {s[:500]}"}
        return {"score": None, "explanation": f"non-JSON judge output: {s[:500]}"}

df_csv = pd.read_csv(CSV_PATH).reset_index().rename(columns={"index": "row_id"})

client = px.Client()
spans_all = client.get_spans_dataframe(project_name=PROJECT)

spans_llm = spans_all[spans_all["name"] == "call_llm"].copy()

output_col = next(
    (c for c in ["attributes.llm.output_messages", "attributes.llm.output_value"] if c in spans_llm.columns),
    None
)
if not output_col:
    raise RuntimeError("No LLM output column found on call_llm spans.")

trace_to_row = _build_trace_to_row_map(spans_all)
spans_llm = spans_llm.merge(trace_to_row, on="context.trace_id", how="left")

if spans_llm["row_id"].isna().all():
    spans_llm["row_id"] = spans_llm[output_col].map(_row_from_answer)

if spans_llm["row_id"].isna().all():
    cols_preview = [c for c in spans_all.columns if "variables" in c.lower() or "prompt" in c.lower()]
    raise RuntimeError(
        "Could not find row_id_for_gt on any span in the trace. "
        f"Inspect possible columns: {cols_preview[:12]}"
    )

spans_llm = spans_llm.merge(df_csv[["row_id", "sample_ground_truth"]], on="row_id", how="left")

df_eval = spans_llm.dropna(subset=["sample_ground_truth"]).set_index("context.span_id")
df_eval = df_eval.rename(columns={output_col: "generated", "sample_ground_truth": "reference"})[
    ["generated", "reference"]
].astype(str)

judge = OpenAIModel(model=JUDGE_MODEL, api_key=OPENAI_API_KEY)

res_groundedness = llm_generate(
    dataframe=df_eval,
    template=TEMPLATE,
    model=judge,
    verbose=False,
    output_parser=_safe_json_parser,
)

res_groundedness.index = df_eval.index
client.log_evaluations(SpanEvaluations(eval_name="Groundedness", dataframe=res_groundedness))
print(f"Logged {len(res_groundedness)} rows for 'Groundedness' to project: {PROJECT}")

def _is_no_answer(txt: str) -> bool:
    t = (txt or "").strip().lower()
    if not t:
        return True
    return any(p in t for p in [
        "i don't know", "i do not know", "cannot answer", "no answer",
        "insufficient context", "not provided in the context", "context does not contain"
    ])

def _score_to_label(score: float, generated: str) -> str:
    if _is_no_answer(generated):
        return "No Answer"
    try:
        s = float(score)
    except Exception:
        return "Neutral"
    if s >= 0.95:
        return "Perfect"
    if s >= 0.80:
        return "Good"
    if s >= 0.50:
        return "Neutral"
    return "Misleading"

rating_df = pd.DataFrame(index=res_groundedness.index)
rating_df["label"] = [_score_to_label(res_groundedness.loc[i, "score"], df_eval.loc[i, "generated"])
                      for i in rating_df.index]
rating_df["explanation"] = [f"Rule-based from groundedness score={res_groundedness.loc[i, 'score']}: "
                            f"{rating_df.loc[i, 'label']}" for i in rating_df.index]

client.log_evaluations(SpanEvaluations(eval_name="Overall Rating", dataframe=rating_df))
print(f"Logged {len(rating_df)} rows for 'Overall Rating' to project: {PROJECT}")

print("\nSample Overall Ratings:")
print(rating_df.head())
