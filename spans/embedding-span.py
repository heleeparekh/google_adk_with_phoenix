from phoenix.otel import register
from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues
import numpy as np

register(project_name="adk-arize-test", auto_instrument=True)
tracer = trace.get_tracer(__name__)

def embed(text):
    with tracer.start_as_current_span("embed_text") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.EMBEDDING.value)
        span.set_attribute("input.value", text)
        # Fake embedding: use random for demo
        embedding = np.random.rand(10).tolist()
        span.set_attribute("output.value", str(embedding))
    return embedding

if __name__ == "__main__":
    emb = embed("Paris is the capital of France.")
    print("Embedding:", emb)
