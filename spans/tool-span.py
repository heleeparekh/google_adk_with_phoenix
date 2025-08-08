from phoenix.otel import register
from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues

import json

register(project_name="adk-arize-test", auto_instrument=True)
tracer = trace.get_tracer(__name__)

def multiply(a, b):
    with tracer.start_as_current_span("multiply_tool") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.TOOL.value)
        # Set tool input as JSON
        span.set_attribute("input.value", json.dumps({"a": a, "b": b}))
        result = a * b
        # Set output
        span.set_attribute("output.value", str(result))
    return result

if __name__ == "__main__":
    res = multiply(7, 6)
    print("Tool result:", res)
