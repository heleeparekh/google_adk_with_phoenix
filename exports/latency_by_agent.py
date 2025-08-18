# latency_error_rate_last24h.py
import os
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import phoenix as px

PROJECT = os.getenv("PHOENIX_PROJECT", "adk-q-and-a")

client = px.Client()
spans = client.get_spans_dataframe(project_name=PROJECT)
spans["start_time"] = pd.to_datetime(spans["start_time"], utc=True, errors="coerce")
spans["end_time"]   = pd.to_datetime(spans["end_time"],   utc=True, errors="coerce")

now = datetime.now(timezone.utc)
spans = spans[(spans["start_time"] >= now - timedelta(hours=24)) & spans["end_time"].notna()]
if spans.empty:
    raise SystemExit("No spans in the last 24h.")

# latency per trace
trace_bounds = spans.groupby("context.trace_id").agg(
    start=("start_time","min"),
    end=("end_time","max")
)
lat_ms = (trace_bounds["end"] - trace_bounds["start"]).dt.total_seconds() * 1000

avg_latency = lat_ms.mean()
p95_latency = np.percentile(lat_ms, 95)

# error rate per trace
# A trace is an error if ANY of its spans has a non-OK status
trace_status = spans.groupby("context.trace_id")["status_code"].apply(
    lambda codes: any(str(c).lower() != "status_code_ok" and str(c).lower() != "ok" for c in codes)
)
error_traces = trace_status.sum()
total_traces = len(trace_status)
error_rate = error_traces / total_traces * 100

print(f"Project: {PROJECT}")
print(f"Traces in last 24h: {total_traces}")
print(f"Average latency: {avg_latency:.2f} ms")
print(f"P95 latency: {p95_latency:.2f} ms")
print(f"Error rate: {error_rate:.2f}%  ({error_traces}/{total_traces} traces failed)")
