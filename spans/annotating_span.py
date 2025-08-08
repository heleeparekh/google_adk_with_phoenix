import time
from phoenix.otel import register
from opentelemetry import trace
from opentelemetry.trace import format_span_id, get_current_span
from phoenix.client import Client

# 1. Register Phoenix tracing
register(project_name="adk-arize-test", auto_instrument=True)
tracer = trace.get_tracer(__name__)

def fake_llm_call(prompt):
    # Dummy answer for demo
    return "42"

def main():
    prompt = "What is the answer to life, the universe, and everything?"
    with tracer.start_as_current_span("llm_with_feedback") as span:
        answer = fake_llm_call(prompt)
        span.set_attribute("input.value", prompt)
        span.set_attribute("output.value", answer)
        print(f"Prompt: {prompt}")
        print(f"LLM answer: {answer}")

        # 2. Get current span id
        current_span = get_current_span()
        span_id = format_span_id(current_span.get_span_context().span_id)
        print(f"Current span_id: {span_id}")

        # 3. Attach a feedback annotation
        client = Client()  # Defaults to http://localhost:6006 if running locally
        annotation = client.annotations.add_span_annotation(
            annotation_name="correctness",    # Must match config in Phoenix project
            annotator_kind="HUMAN",           # or "LLM" or "CODE"
            span_id=span_id,
            label="Correct",                  # or whatever label you use
            score=1,                          # e.g. 1 for correct, 0 for incorrect, or any float
            explanation="The response is correct according to the user.",
            metadata={"user_id": "user-123"},
        )
        print("Annotation sent to Phoenix:", annotation)

if __name__ == "__main__":
    main()
