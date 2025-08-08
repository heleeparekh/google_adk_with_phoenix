from phoenix.otel import register
from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes
from openinference.semconv.trace import OpenInferenceSpanKindValues
import os
from dotenv import load_dotenv
load_dotenv()

register(project_name="adk-arize-test", auto_instrument=True)
tracer = trace.get_tracer(__name__)

import google.generativeai as genai
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

with tracer.start_as_current_span("llm_generate_content") as span:
    # Set the correct span kind using OpenInference convention
    span.set_attribute(
        SpanAttributes.OPENINFERENCE_SPAN_KIND,
        OpenInferenceSpanKindValues.LLM.value  # --> Correct span kind “LLM”
    )
    prompt = "What is the capital of France?"
    span.set_attribute("input.value", prompt)
    response = model.generate_content(prompt)
    span.set_attribute("output.value", response.text)
