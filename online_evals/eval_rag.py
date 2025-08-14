# eval_rag.py — minimal RAG evals (Context Relevancy + Answer Faithfulness)

import os
import ast
import pandas as pd
import phoenix as px
from dotenv import load_dotenv
from phoenix.trace import SpanEvaluations
from phoenix.evals import OpenAIModel, llm_classify, RAG_RELEVANCY_PROMPT_TEMPLATE

load_dotenv()

PROJECT = os.getenv("PHOENIX_PROJECT", "adk-rag-csv")
JUDGE = OpenAIModel(model=os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o"),
                    api_key=os.getenv("OPENAI_API_KEY"))

client = px.Client()

# --- compact but ADK-aware flattener for output_messages ---
def flatten_msgs(x):
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, list):
        parts = []
        for m in x:
            if not isinstance(m, dict):
                parts.append(str(m)); continue
            # common keys:
            if "message.content" in m and m["message.content"]:
                parts.append(str(m["message.content"]))
            elif "message.contents" in m and isinstance(m["message.contents"], list):
                for part in m["message.contents"]:
                    if isinstance(part, dict):
                        if "message_content.text" in part and part["message_content.text"]:
                            parts.append(str(part["message_content.text"]))
                        elif "text" in part and part["text"]:
                            parts.append(str(part["text"]))
            elif "content" in m and m["content"]:
                parts.append(str(m["content"]))
            elif "message" in m and isinstance(m["message"], dict) and m["message"].get("content"):
                parts.append(str(m["message"]["content"]))
        return " ".join(p for p in parts if p).strip()
    return "" if x is None else str(x)

# --- fetch spans & required fields ---
spans = client.get_spans_dataframe(project_name=PROJECT)
spans = spans[spans["name"] == "call_llm"][[
    "context.span_id",
    "attributes.llm.prompt_template.variables",
    "attributes.llm.output_messages",
]].copy()
if spans.empty:
    raise SystemExit("No 'call_llm' spans found. Run your agent first.")

# pull {query, context} and answer text
def get_var(obj, key):
    if isinstance(obj, str):
        try:
            obj = ast.literal_eval(obj)
        except Exception:
            return ""
    return (obj or {}).get(key, "") if isinstance(obj, dict) else ""

spans["query"] = spans["attributes.llm.prompt_template.variables"].apply(lambda v: str(get_var(v, "query")).strip())
spans["context"] = spans["attributes.llm.prompt_template.variables"].apply(lambda v: str(get_var(v, "context")).strip())
spans["answer"] = spans["attributes.llm.output_messages"].apply(flatten_msgs)

# ---------- Eval A: Context Relevancy (query vs context) ----------
ctx_df = spans.loc[(spans["query"] != "") & (spans["context"] != ""), ["context.span_id"]].copy()
if not ctx_df.empty:
    ctx_df["input"] = spans.loc[ctx_df.index, "query"].values
    ctx_df["reference"] = spans.loc[ctx_df.index, "context"].values
    ctx_df = ctx_df.set_index("context.span_id")

    ctx_results = llm_classify(
        data=ctx_df,
        model=JUDGE,
        template=RAG_RELEVANCY_PROMPT_TEMPLATE,
        rails=["relevant", "unrelated"],
        provide_explanation=True,
    )
    ctx_results.index = ctx_df.index
    ctx_results.index.name = "context.span_id"
    ctx_results["score"] = (ctx_results["label"].str.lower() == "relevant").astype(int)

    client.log_evaluations(SpanEvaluations("RAG Context Relevancy", ctx_results))
    print(f"Logged {len(ctx_results)} rows for 'RAG Context Relevancy'.")
else:
    print("No rows for Context Relevancy (missing query/context).")

# ---------- Eval B: Answer Faithfulness (answer vs same context) ----------
faith_df = spans.loc[(spans["answer"] != "") & (spans["context"] != ""), ["context.span_id"]].copy()
if not faith_df.empty:
    faith_df["input"] = spans.loc[faith_df.index, "answer"].values      # ANSWER
    faith_df["reference"] = spans.loc[faith_df.index, "context"].values # CONTEXT
    faith_df = faith_df.set_index("context.span_id")

    FAITHFULNESS_TEMPLATE = (
        "You are a strict evaluator.\n"
        "Given CONTEXT and ANSWER, determine if the ANSWER is supported by the CONTEXT.\n"
        'Respond with exactly one label from: grounded, hallucinated.\n\n'
        "CONTEXT:\n{reference}\n\nANSWER:\n{input}\n"
    )

    faith_results = llm_classify(
        data=faith_df,
        model=JUDGE,
        template=FAITHFULNESS_TEMPLATE,
        rails=["grounded", "hallucinated"],
        provide_explanation=True,
    )
    faith_results.index = faith_df.index
    faith_results.index.name = "context.span_id"
    faith_results["score"] = (faith_results["label"].str.lower() == "grounded").astype(int)

    client.log_evaluations(SpanEvaluations("RAG Answer Faithfulness", faith_results))
    print(f"Logged {len(faith_results)} rows for 'RAG Answer Faithfulness'.")
else:
    print("No rows for Answer Faithfulness (missing answer/context).")
