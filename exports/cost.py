# latency_error_cost_from_tokens_last24h.py
import os
from datetime import datetime, timedelta, timezone
import re
import numpy as np
import pandas as pd
import phoenix as px

PROJECT = os.getenv("PHOENIX_PROJECT", "adk-q-and-a")

# Prices in USD per 1,000,000 tokens. Edit to match your billing.
# You can add more entries; pattern is a case-insensitive regex matched against attributes.llm.model_name
PRICES = [
    {"pattern": r"gemini-2\.0-flash", "in_per_m": 0.075, "out_per_m": 0.30},   # adjust if your account differs
    {"pattern": r"gemini-1\.5-flash", "in_per_m": 0.075, "out_per_m": 0.30},
    {"pattern": r"gpt-4o-mini",       "in_per_m": 0.50,  "out_per_m": 1.50},
    {"pattern": r"gpt-4o",            "in_per_m": 5.00,  "out_per_m": 15.00},
    {"pattern": r"claude-3\.5",       "in_per_m": 3.00,  "out_per_m": 15.00},
]

def price_for(model_name: str):
    if not isinstance(model_name, str):
        return None
    for row in PRICES:
        if re.search(row["pattern"], model_name, flags=re.IGNORECASE):
            return row["in_per_m"], row["out_per_m"]
    return None

client = px.Client()
spans = client.get_spans_dataframe(project_name=PROJECT)

# parse times and filter to last 24h
spans["start_time"] = pd.to_datetime(spans["start_time"], utc=True, errors="coerce")
spans["end_time"]   = pd.to_datetime(spans["end_time"],   utc=True, errors="coerce")
now = datetime.now(timezone.utc)
spans = spans[(spans["start_time"] >= now - timedelta(hours=24)) & spans["end_time"].notna()]
if spans.empty:
    raise SystemExit("No spans in the last 24h.")

# latency from trace envelope
trace_bounds = spans.groupby("context.trace_id").agg(
    start=("start_time","min"),
    end=("end_time","max")
)
lat_ms = (trace_bounds["end"] - trace_bounds["start"]).dt.total_seconds() * 1000
avg_latency = float(lat_ms.mean())
p95_latency = float(np.percentile(lat_ms, 95))

# error rate: a trace is error if any span has non-OK status
trace_has_error = spans.groupby("context.trace_id")["status_code"].apply(
    lambda codes: any(str(c).lower() not in ("status_code_ok", "ok") for c in codes)
)
total_traces = int(trace_has_error.shape[0])
error_traces = int(trace_has_error.sum())
error_rate = 100.0 * error_traces / total_traces if total_traces else 0.0

# cost from token counts
TOK_PROMPT = "attributes.llm.token_count.prompt"
TOK_COMP   = "attributes.llm.token_count.completion"
MODEL_COL  = "attributes.llm.model_name"

have_tokens = all(col in spans.columns for col in [TOK_PROMPT, TOK_COMP, MODEL_COL])
if not have_tokens:
    raise SystemExit("Token/model columns not found; expected attributes.llm.token_count.prompt, .completion, and .model_name")

# keep only rows with a model_name and any token counts
llm_rows = spans[spans[MODEL_COL].notna()].copy()
llm_rows[TOK_PROMPT] = llm_rows[TOK_PROMPT].fillna(0)
llm_rows[TOK_COMP]   = llm_rows[TOK_COMP].fillna(0)

# map model -> prices
prices = llm_rows[MODEL_COL].apply(price_for)
llm_rows["in_per_m"]  = prices.apply(lambda p: p[0] if p else np.nan)
llm_rows["out_per_m"] = prices.apply(lambda p: p[1] if p else np.nan)

# drop rows with unknown pricing
llm_rows = llm_rows.dropna(subset=["in_per_m", "out_per_m"])

# span cost in USD
llm_rows["span_cost_usd"] = (
    (llm_rows[TOK_PROMPT] / 1_000_000.0) * llm_rows["in_per_m"]
    + (llm_rows[TOK_COMP] / 1_000_000.0) * llm_rows["out_per_m"]
)

# sum to trace cost
trace_costs = llm_rows.groupby("context.trace_id")["span_cost_usd"].sum(min_count=1)

avg_cost   = float(trace_costs.mean()) if not trace_costs.empty else 0.0
total_cost = float(trace_costs.sum())  if not trace_costs.empty else 0.0

print(f"Project: {PROJECT}")
print(f"Traces in last 24h: {total_traces}")
print(f"Average latency: {avg_latency:.2f} ms")
print(f"P95 latency: {p95_latency:.2f} ms")
print(f"Error rate: {error_rate:.2f}%  ({error_traces}/{total_traces} traces failed)")
print(f"Average cost per trace: ${avg_cost:.4f}")
print(f"Total cost (last 24h): ${total_cost:.4f}")

# optional: show a few per-trace costs
out = trace_costs.reset_index().rename(columns={"span_cost_usd": "trace_cost_usd"})
print("\nSample per-trace costs:")
print(out.head(10).to_string(index=False))
