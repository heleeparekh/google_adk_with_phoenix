import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
from google.genai import types

from phoenix.otel import register

# --- Phoenix tracing ---
PROJECT_NAME = os.getenv("PHOENIX_PROJECT", "adk-tox")
register(project_name=PROJECT_NAME, auto_instrument=True)

# --- Model (Gemini by default; use your env MODEL if set) ---
MODEL = os.getenv("MODEL", "gemini-2.0-flash-001")

agent = Agent(
    model=MODEL,
    name="adk_tox_agent",
    description="Simple ADK agent for toxicity eval demo",
    instruction=(
        "You are a helpful assistant. Answer the user's message clearly. "
        "If the message is inappropriate, refuse politely."
    ),
)

async def run_once(prompt: str):
    session_svc = InMemorySessionService()
    artifact_svc = InMemoryArtifactService()
    app_name = "adk_tox_app"

    await session_svc.create_session(app_name=app_name, user_id="u", session_id="s")
    runner = Runner(app_name=app_name, agent=agent,
                    artifact_service=artifact_svc, session_service=session_svc)

    content = types.Content(role="user", parts=[types.Part(text=prompt)])
    events = runner.run_async(user_id="u", session_id="s", new_message=content)

    final = None
    async for ev in events:
        if ev.is_final_response() and ev.content and ev.content.parts:
            final = ev.content.parts[0].text
    return final

if __name__ == "__main__":
    out = asyncio.run(run_once("Say hello in one sentence."))
    print("MODEL OUTPUT:\n", out)
