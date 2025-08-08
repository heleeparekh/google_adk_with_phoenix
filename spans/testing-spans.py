import json
from datetime import datetime, timedelta
import numpy as np
from opentelemetry import trace
from phoenix.otel import register
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

# Setup LLM
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm_model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

register(project_name="adk-arize-test", auto_instrument=True)
tracer = trace.get_tracer(__name__)

# Simple KBs
IT_KB = [
    {"question": "reset password", "answer": "To reset your password, visit the IT portal and follow the reset link."},
    {"question": "vpn", "answer": "For VPN access, install the VPN client and use your company credentials."}
]
HR_KB = [
    {"question": "vacation days", "answer": "You have 18 vacation days remaining for this year."},
    {"question": "pay slip", "answer": "Download your latest pay slip from the HR portal."}
]

def classify_question(question):
    with tracer.start_as_current_span("classifier") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.AGENT.value)  # OR use AGENT or TOOL, whichever is closer semantically
        span.set_attribute("span.subkind", "CLASSIFIER")
    # with tracer.start_as_current_span("classifier") as span:
    #     span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "CLASSIFIER")
        span.set_attribute("input.value", question)
        # Basic keyword classification
        if any(word in question.lower() for word in ["password", "vpn", "computer", "system", "email"]):
            label = "IT"
        elif any(word in question.lower() for word in ["vacation", "leave", "pay slip", "salary", "holiday"]):
            label = "HR"
        else:
            label = "HR"  # default/fallback
        span.set_attribute("output.value", label)
        return label

def embed_question(question):
    with tracer.start_as_current_span("embedding") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.EMBEDDING.value)
        span.set_attribute("input.value", question)
        embedding = np.random.rand(10).tolist()  # Dummy embedding
        span.set_attribute("output.value", json.dumps(embedding))
    return embedding

def rag_retrieve(label, question_embedding):
    with tracer.start_as_current_span("rag_retriever") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.RETRIEVER.value)
        span.set_attribute("input.value", f"{label} KB, embedding")
        kb = IT_KB if label == "IT" else HR_KB
        # Simulate retrieval: pick doc with keyword overlap
        for doc in kb:
            if any(word in doc["question"] for word in question_embedding):
                result = doc["answer"]
                span.set_attribute("output.value", result)
                return result
        # If no overlap, return first doc
        result = kb[0]["answer"]
        span.set_attribute("output.value", result)
        return result

def datetime_tool():
    with tracer.start_as_current_span("datetime_tool") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.TOOL.value)
        now = datetime.now()
        next_bd = now + timedelta(days=1 if now.weekday() < 4 else 7 - now.weekday())
        result = next_bd.strftime("%A, %B %d")
        span.set_attribute("input.value", now.isoformat())
        span.set_attribute("output.value", result)
    return result

def answer_query(user_query):
    with tracer.start_as_current_span("agent_handler") as agent_span:
        agent_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "AGENT")
        agent_span.set_attribute("input.value", user_query)

        label = classify_question(user_query)
        embedding = embed_question(user_query)
        # For demo: create "fake tokens" from question as embedding search
        fake_embedding_tokens = [word for word in user_query.lower().split() if len(word) > 2]

        context = rag_retrieve(label, fake_embedding_tokens)
        response_by = datetime_tool()
        with tracer.start_as_current_span("compose_llm_response") as span:
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value)
            llm_prompt = (
                f"User asked: '{user_query}'\n"
                f"Topic: {label}\n"
                f"Relevant Info: {context}\n"
                f"Reply: Use the info and tell user their query will be resolved by {response_by}."
            )
            span.set_attribute("input.value", llm_prompt)
            llm_response = llm_model.generate_content(llm_prompt)
            span.set_attribute("output.value", llm_response.text)
        agent_span.set_attribute("output.value", llm_response.text)
        return llm_response.text

if __name__ == "__main__":
    print("\nQ1: How do I reset my password?")
    print(answer_query("How do I reset my password?"))
    print("\nQ2: How many vacation days do I have?")
    print(answer_query("How many vacation days do I have?"))
    print("\nQ3: How can I access VPN?")
    print(answer_query("How can I access VPN?"))
    print("\nQ4: Where do I download my pay slip?")
    print(answer_query("Where do I download my pay slip?"))
