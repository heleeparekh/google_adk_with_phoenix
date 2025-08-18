# eval_fluency_min.py
import os, json, pandas as pd, phoenix as px
from dotenv import load_dotenv
from phoenix.trace import SpanEvaluations
from phoenix.evals import OpenAIModel, llm_generate

load_dotenv()
PROJECT = os.getenv("PHOENIX_PROJECT", "adk-q-and-a")
JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

TEMPLATE = """
Return only a JSON object with keys "score" and "evidence". No prose, no code fences.

Task: Evaluate the English grammar, fluency, clarity, and readability of the Generated Answer.
Ignore factual correctness or domain accuracy. Focus purely on writing quality.

Scoring guide:
1.00 flawless or nearly flawless
0.90 excellent with at most one very minor issue
0.75 good with a few minor issues
0.50 fair with multiple issues affecting clarity
0.25 poor with frequent errors or awkward phrasing
0.00 incomprehensible, empty, or non-English

Generated Answer:
{generated}
"""

def main():
    client = px.Client()
    spans = client.get_spans_dataframe(project_name=PROJECT)
    spans = spans[spans["name"] == "call_llm"].copy()
    if spans.empty:
        print("No call_llm spans found.")
        return

    out_col = "attributes.llm.output_messages" if "attributes.llm.output_messages" in spans.columns else "attributes.llm.output_value"
    if out_col not in spans.columns:
        raise RuntimeError("No LLM output column found on call_llm spans.")

    df_eval = spans.set_index("context.span_id")[[out_col]].rename(columns={out_col: "generated"})
    df_eval["generated"] = df_eval["generated"].astype(str)
    df_eval = df_eval[df_eval["generated"].str.strip().astype(bool)]
    if df_eval.empty:
        print("No non-empty generated answers to evaluate.")
        return

    judge = OpenAIModel(model=JUDGE_MODEL, api_key=OPENAI_API_KEY)

    res = llm_generate(
        dataframe=df_eval[["generated"]],
        template=TEMPLATE,
        model=judge,
        verbose=False,
        output_parser=lambda out, _: json.loads(out),
    )
    res.index = df_eval.index

    # Numeric metric
    fluency_df = pd.DataFrame(index=res.index, data={"score": res["score"]})
    client.log_evaluations(SpanEvaluations(eval_name="Fluency", dataframe=fluency_df))

    # Textual justification: must be under the column name 'explanation'
    evidence_df = pd.DataFrame(index=res.index)
    evidence_df["explanation"] = res["evidence"].astype(str).fillna("")
    client.log_evaluations(SpanEvaluations(eval_name="Fluency Evidence", dataframe=evidence_df))

    print(f"Logged {len(fluency_df)} rows for Fluency and {len(evidence_df)} for Fluency Evidence to project: {PROJECT}")

if __name__ == "__main__":
    main()
