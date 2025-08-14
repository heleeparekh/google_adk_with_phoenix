# eval_toxicity_min.py
import os
import pandas as pd
import phoenix as px
from dotenv import load_dotenv
from phoenix.trace import SpanEvaluations
from phoenix.evals import OpenAIModel, TOXICITY_PROMPT_TEMPLATE, TOXICITY_PROMPT_RAILS_MAP, llm_classify

load_dotenv()
PROJECT = os.getenv("PHOENIX_PROJECT", "adk-tox")
judge = OpenAIModel(model=os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o"),
                    api_key=os.getenv("OPENAI_API_KEY"))

client = px.Client()
spans = client.get_spans_dataframe(project_name=PROJECT)
spans = spans[spans["name"] == "call_llm"][["context.span_id", "attributes.llm.output_messages"]]

# Build dataframe expected by the template: index=span_id, column 'input'
df = pd.DataFrame({
    "context.span_id": spans["context.span_id"],
    "input": spans["attributes.llm.output_messages"].astype(str),
}).set_index("context.span_id")

rails = list(TOXICITY_PROMPT_RAILS_MAP.values())
res = llm_classify(
    data=df,
    model=judge,
    template=TOXICITY_PROMPT_TEMPLATE,  # your install expects {input}
    rails=rails,
    provide_explanation=True,
)

# Attach back to spans and add a numeric score (optional)
res.index = df.index
res.index.name = "context.span_id"
res["score"] = res["label"].str.lower().map({"toxic": 1, "non-toxic": 0}).fillna(0).astype(int)

client.log_evaluations(SpanEvaluations(eval_name="Toxicity", dataframe=res))
print(f"Logged {len(res)} rows for 'Toxicity' to project: {PROJECT}")
