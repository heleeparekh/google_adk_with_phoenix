import os
import asyncio
from dotenv import load_dotenv

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
from google.genai import types

from phoenix.otel import register
from openinference.instrumentation import using_prompt_template

load_dotenv()

# Register tracing to Phoenix for this app
register(project_name="adk-agent-evals", auto_instrument=True)

MODEL = os.getenv("MODEL", "gemini-2.0-flash-001")
APP_NAME = "adk_eval_demo"

# A simple single agent that answers questions with code snippets when asked
instruction = """
You are a helpful assistant. If the user asks for code, return only valid code inside triple backticks.
If the user asks a factual or descriptive question, answer succinctly.
"""

agent = Agent(
    model=MODEL,
    name="adk_simple_agent",
    description="Simple ADK agent for evaluation demo",
    instruction=instruction,
    generate_content_config=types.GenerateContentConfig(temperature=0.2),
)

session_service = InMemorySessionService()
artifact_service = InMemoryArtifactService()

async def run_once(query: str, user_id="user1", session_id="session1"):
    session = await session_service.create_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
    runner = Runner(app_name=APP_NAME, agent=agent, artifact_service=artifact_service, session_service=session_service)

    # Add prompt template metadata to the active span
    with using_prompt_template(template="{query}", variables={"query": query}, version="v1"):
        content = types.Content(role="user", parts=[types.Part(text=query)])
        events = runner.run_async(user_id=user_id, session_id=session_id, new_message=content)

        final = None
        async for event in events:
            if event.is_final_response() and event.content and event.content.parts:
                final = event.content.parts[0].text
        return final

if __name__ == "__main__":
    # Example query that produces code so we can evaluate later
    out = asyncio.run(run_once("Write a Python function factorial(n) using iteration."))
    print("Final response:\n", out)
