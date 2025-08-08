import os
from dotenv import load_dotenv
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from openai import OpenAI

load_dotenv()  # Load environment variables from .env

tracer_provider = register(project_name="adk-arize-test")
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider, skip_dep_check=True)

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image?"},
                {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"}},
            ],
        }
    ],
    max_tokens=300,
)
print(response.choices[0].message.content)