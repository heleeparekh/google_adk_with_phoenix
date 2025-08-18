# export_spans_with_annotations.py
# Exports last 24h spans + annotations from a running Phoenix APP (no /collector).
# Only saves CSV (no Parquet).

import os
from datetime import datetime, timedelta, timezone
import pandas as pd

APP_ENDPOINT = os.getenv("PHOENIX_APP_ENDPOINT", "http://127.0.0.1:6006")
PROJECT_NAME = os.getenv("PHOENIX_PROJECT", "adk-q-and-a")
CLIENT_HEADERS = os.getenv("PHOENIX_CLIENT_HEADERS")

os.environ["PHOENIX_SESSION_ENDPOINT"] = APP_ENDPOINT
if CLIENT_HEADERS:
    os.environ["PHOENIX_CLIENT_HEADERS"] = CLIENT_HEADERS
else:
    os.environ.pop("PHOENIX_CLIENT_HEADERS", None)

try:
    from phoenix.client import Client as PhoenixClient
except Exception as e:
    raise RuntimeError(
        "phoenix.client.Client is not available. "
        "Upgrade the phoenix package to a version that supports annotated spans."
    ) from e

def to_utc(ts):
    if pd.api.types.is_datetime64_any_dtype(ts):
        return ts.dt.tz_convert("UTC") if ts.dt.tz is not None else ts.dt.tz_localize("UTC")
    return pd.to_datetime(ts, utc=True, errors="coerce")

def main():
    client = PhoenixClient()

    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(hours=24)

    # Pull all spans
    spans = client.spans.get_spans_dataframe(project_identifier=PROJECT_NAME)
    if spans is None or len(spans) == 0:
        print(f"No spans found for project '{PROJECT_NAME}'.")
        return

    # Time filter
    if "start_time" in spans.columns:
        spans["start_time"] = to_utc(spans["start_time"])
        mask = spans["start_time"].between(start_utc, now_utc)
    elif "end_time" in spans.columns:
        spans["end_time"] = to_utc(spans["end_time"])
        mask = spans["end_time"].between(start_utc, now_utc)
    else:
        mask = pd.Series(True, index=spans.index)

    spans_24h = spans[mask].copy()
    if len(spans_24h) == 0:
        print(f"No spans in last 24h for project '{PROJECT_NAME}'.")
        return

    # Fetch annotations
    annotations = client.spans.get_span_annotations_dataframe(
        span_ids=list(spans_24h.index),
        project_identifier=PROJECT_NAME,
    )

    ts = now_utc.strftime("%Y%m%d_%H%M%S")

    # Save spans
    spans_csv = f"spans_{PROJECT_NAME}_{ts}.csv"
    spans_24h.to_csv(spans_csv, index=False)

    if annotations is None or len(annotations) == 0:
        print(f"Exported {len(spans_24h)} spans but no annotations were found.")
        print(f"Spans CSV: {spans_csv}")
        return

    # Save annotations
    ann_csv = f"annotations_{PROJECT_NAME}_{ts}.csv"
    annotations.to_csv(ann_csv, index=False)

    # Join spans + annotations
    joined = annotations.join(spans_24h, how="left")
    joined_csv = f"spans_with_annotations_{PROJECT_NAME}_{ts}.csv"
    joined.to_csv(joined_csv, index=False)

    print(f"Exported {len(spans_24h)} spans and {len(annotations)} annotations from '{PROJECT_NAME}'.")
    print(f"Spans CSV: {spans_csv}")
    print(f"Annotations CSV: {ann_csv}")
    print(f"Joined CSV: {joined_csv}")

if __name__ == "__main__":
    main()
