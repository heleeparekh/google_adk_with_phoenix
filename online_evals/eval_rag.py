# eval_rag.py — evaluate ADK RAG agent from LLM spans (local Phoenix)

import os
import ast
import json
import pandas as pd
import phoenix as px
from phoenix.trace import SpanEvaluations
from phoenix.evals import llm_classify, OpenAIModel, RAG_RELEVANCY_PROMPT_TEMPLATE
from dotenv import load_dotenv

# ---------------------------
# Setup
# ---------------------------
load_dotenv()

PROJECT = os.getenv("PHOENIX_PROJECT", "adk-rag-csv")
OPENAI_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise SystemExit("OPENAI_API_KEY not set. Put it in .env or export it, then rerun.")

client = px.Client()              # local Phoenix HTTP API
judge = OpenAIModel(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)

# ---------------------------
# Helpers
# ---------------------------
def _safe_parse_obj(raw):
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        for loader in (json.loads, ast.literal_eval):
            try:
                return loader(raw)
            except Exception:
                pass
        return raw
    return raw

def parse_msgs(raw) -> str:
    val = _safe_parse_obj(raw)
    if isinstance(val, list):
        out = []
        for m in val:
            if isinstance(m, dict):
                if "message.content" in m:
                    out.append(str(m["message.content"]))
                elif "message.contents" in m and isinstance(m["message.contents"], list):
                    for part in m["message.contents"]:
                        if isinstance(part, dict) and "message_content.text" in part:
                            out.append(str(part["message_content.text"]))
                elif "content" in m:
                    out.append(str(m["content"]))
            elif isinstance(m, str):
                out.append(m)
        return " ".join(out).strip()
    if isinstance(val, str):
        return val.strip()
    return "" if val is None else str(val)

# ---------------------------
# Pull LLM spans and required columns directly
# ---------------------------
spans = client.get_spans_dataframe(project_name=PROJECT)
if spans.empty:
    raise SystemExit(f"No spans found for project '{PROJECT}'. Run rag_agent.py first.")

# Keep only the LLM spans
if "name" in spans.columns:
    spans = spans[spans["name"] == "call_llm"].copy()
if spans.empty:
    raise SystemExit("No 'call_llm' spans found. Verify tracing and project name.")

needed = [
    "context.span_id",
    "attributes.llm.prompt_template.variables",
    "attributes.llm.output_messages",
]
missing = [c for c in needed if c not in spans.columns]
if missing:
    raise SystemExit(f"Missing columns on spans: {missing}")

# Build rows for the two evals from the prompt-template variables + model output
rows_ctx, rows_faith = [], []

for _, r in spans.iterrows():
    span_id = r["context.span_id"]
    vars_obj = _safe_parse_obj(r.get("attributes.llm.prompt_template.variables"))
    query_text = ""
    context_text = ""
    if isinstance(vars_obj, dict):
        qv = vars_obj.get("query")
        cv = vars_obj.get("context")
        if isinstance(qv, str):
            query_text = qv.strip()
        if isinstance(cv, str):
            context_text = cv.strip()

    answer_text = parse_msgs(r.get("attributes.llm.output_messages"))

    if query_text and context_text:
        rows_ctx.append({
            "context.span_id": span_id,
            "input": query_text,
            "reference": context_text,
        })
    if answer_text and context_text:
        rows_faith.append({
            "context.span_id": span_id,
            "input": answer_text,      # ANSWER
            "reference": context_text, # CONTEXT
        })

# ---------------------------
# Eval A: RAG Context Relevancy (query ↔ context)
# ---------------------------
if rows_ctx:
    ctx_df = pd.DataFrame(rows_ctx).set_index("context.span_id")
    ctx_results = llm_classify(
        data=ctx_df,
        model=judge,
        template=RAG_RELEVANCY_PROMPT_TEMPLATE,
        rails=["relevant", "unrelated"],
        concurrency=8,
        provide_explanation=True,
    )
    ctx_results["score"] = (ctx_results["label"].str.lower() == "relevant").astype(int)
    client.log_evaluations(SpanEvaluations(
        eval_name="RAG Context Relevancy",
        dataframe=ctx_results
    ))
    print(f"Logged {len(ctx_results)} rows for 'RAG Context Relevancy'.")
else:
    print("No rows for Context Relevancy (missing query/context variables).")

# ---------------------------
# Eval B: RAG Answer Faithfulness (answer ↔ context)
# ---------------------------
FAITHFULNESS_TEMPLATE = """You are a strict evaluator.
Given CONTEXT and ANSWER, determine if the ANSWER is supported by the CONTEXT.
Respond with exactly one label from: grounded, hallucinated.
Only consider information present in CONTEXT; do not assume external knowledge.

CONTEXT:
{reference}

ANSWER:
{input}
"""

if rows_faith:
    faith_df = pd.DataFrame(rows_faith).set_index("context.span_id")
    faith_results = llm_classify(
        data=faith_df,
        model=judge,
        template=FAITHFULNESS_TEMPLATE,
        rails=["grounded", "hallucinated"],
        concurrency=8,
        provide_explanation=True,
    )
    faith_results["score"] = (faith_results["label"].str.lower() == "grounded").astype(int)
    client.log_evaluations(SpanEvaluations(
        eval_name="RAG Answer Faithfulness",
        dataframe=faith_results
    ))
    print(f"Logged {len(faith_results)} rows for 'RAG Answer Faithfulness'.")
else:
    print("No rows for Answer Faithfulness (missing answer or context).")

print("Done. Open Phoenix (local UI) → project:", PROJECT)
print("Open a 'call_llm' span → Annotations, or use the top-level Evaluations view.")
