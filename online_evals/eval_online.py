# eval_online.py — Local Phoenix + OpenAI judge

import os
import ast
import pandas as pd
import phoenix as px
from phoenix.trace import SpanEvaluations
from phoenix.evals import llm_classify, OpenAIModel, RAG_RELEVANCY_PROMPT_TEMPLATE
from dotenv import load_dotenv

# 1) Load .env (ensure you run from the folder that contains .env, or pass a path)
load_dotenv()

PROJECT = os.getenv("PHOENIX_PROJECT", "adk-agent-evals")
OPENAI_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o")  # model name comes from its own var
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # key from .env

if not OPENAI_API_KEY:
    raise SystemExit("OPENAI_API_KEY not found. Put 'OPENAI_API_KEY=sk-...' in .env or export it before running.")

# 2) Connect to local Phoenix HTTP API (no base_url arg on older clients)
client = px.Client()

# 3) Pull spans
spans_df = client.get_spans_dataframe(project_name=PROJECT)
if spans_df.empty:
    raise SystemExit(f"No spans found for project '{PROJECT}'. Run the agent and try again.")

# Keep only the LLM call spans
if "name" in spans_df.columns:
    spans_df = spans_df[spans_df["name"] == "call_llm"].copy()
if spans_df.empty:
    raise SystemExit("No 'call_llm' spans found. Verify tracing and project name.")

required_cols = [
    "context.span_id",
    "attributes.llm.input_messages",
    "attributes.llm.output_messages",
]
missing = [c for c in required_cols if c not in spans_df.columns]
if missing:
    raise SystemExit(f"Missing expected columns: {missing}. Inspect spans_df.columns and adjust mapping.")

def parse_msgs(raw):
    if raw is None or raw == "":
        return ""
    val = raw
    if isinstance(raw, str):
        try:
            val = ast.literal_eval(raw)  # your columns are Python-literal strings
        except Exception:
            return raw.strip()
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
    return str(val)

df = spans_df.loc[:, required_cols].copy()
df["input"] = df["attributes.llm.input_messages"].apply(parse_msgs)
df["reference"] = df["attributes.llm.output_messages"].apply(parse_msgs)

eval_df = df.loc[df["reference"].str.len() > 0, ["context.span_id", "input", "reference"]].copy()
if eval_df.empty:
    raise SystemExit("No evaluable rows. Model output was empty after parsing.")
eval_df = eval_df.set_index("context.span_id")

# 4) LLM-as-judge with OpenAI
model = OpenAIModel(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)

results = llm_classify(
data=eval_df,
model=model,
template=RAG_RELEVANCY_PROMPT_TEMPLATE,
rails=["relevant", "unrelated"],
concurrency=8,
provide_explanation=True,
)

def to_score(expl):
    if not isinstance(expl, str):
        return 0
    return 1 if "relevant" in expl.lower() else 0

results["score"] = (results["label"].str.lower() == "relevant").astype(int)

# 5) Log back to Phoenix
client.log_evaluations(SpanEvaluations(eval_name="ADK Response Relevancy", dataframe=results))

print(f"Logged {len(results)} rows under eval name: ADK Response Relevancy")
print(f"Project: {PROJECT}")
