from phoenix.otel import register
from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues

register(project_name="adk-arize-test", auto_instrument=True)
tracer = trace.get_tracer(__name__)

def retrieve(query):
    with tracer.start_as_current_span("simple_retriever") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.RETRIEVER.value)
        span.set_attribute("input.value", query)
        # Fake docs
        docs = [
            {"title": "Paris", "content": "Paris is the capital of France."},
            {"title": "Berlin", "content": "Berlin is the capital of Germany."}
        ]
        # Retrieve documents matching query
        results = [doc for doc in docs if query.lower() in doc["content"].lower()]
        for idx, doc in enumerate(results):
            # Add each retrieved doc as attribute
            span.set_attribute(f"retrieval.documents.{idx}.document.content", doc["content"])
        span.set_attribute("output.value", str(results))
    return results

if __name__ == "__main__":
    retrieved = retrieve("capital of France")
    print("Retrieved docs:", retrieved)
