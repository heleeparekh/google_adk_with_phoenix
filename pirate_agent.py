import os
import time
import asyncio

# Import libraries from the Agent Framework
from google.adk.agents import Agent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from phoenix.otel import register
from dotenv import load_dotenv
load_dotenv()

# Get the model ID from the environment variable
tracer_provider = register(
    project_name="adk-arize-test",  # Use any project name you like
    auto_instrument=True
)
MODEL = os.getenv("MODEL", "gemini-2.0-flash") # The model ID for the agent
AGENT_APP_NAME = 'agent_pirate_translator'

# Create InMemory services for session and artifact management
session_service = InMemorySessionService()
artifact_service = InMemoryArtifactService()

async def send_query_to_agent(agent, query, user_id="user", session_id="user_session"):
    """Sends a query to the specified agent and prints the response.

    Args:
        agent: The agent to send the query to.
        query: The query to send to the agent.

    Returns:
        A tuple containing the elapsed time (in milliseconds) and the final response from the agent.
    """

    # Create a new session - if you want to keep the history of interaction you need to move the 
    # creation of the session outside of this function. Here we create a new session per query
    session = await session_service.create_session(app_name=AGENT_APP_NAME,
                                                   user_id=user_id,
                                                   session_id=session_id)
    # Create a content object representing the user's query
    print('\nUser Query: ', query)
    content = types.Content(role='user', parts=[types.Part(text=query)])

    # Start a timer to measure the response time
    start_time = time.time()

    # Create a runner object to manage the interaction with the agent
    runner = Runner(app_name=AGENT_APP_NAME, agent=agent, artifact_service=artifact_service, session_service=session_service)

    # Run the interaction with the agent and get a stream of events
    events = runner.run_async(user_id=user_id, session_id=session_id, new_message=content)

    final_response = None
    elapsed_time_ms = 0.0

    # Loop through the events returned by the runner
    async for event in events:

        is_final_response = event.is_final_response()

        if not event.content:
             continue

        if is_final_response:
            end_time = time.time()
            elapsed_time_ms = round((end_time - start_time) * 1000, 3)

            print("-----------------------------")
            print('>>> Inside final response <<<')
            print("-----------------------------")
            final_response = event.content.parts[0].text # Get the final response from the agent
            print(f'Agent: {event.author}')
            print(f'Response time: {elapsed_time_ms} ms\n')
            print(f'Final Response:\n{final_response}')
            print("----------------------------------------------------------\n")

    return elapsed_time_ms, final_response

if __name__ == '__main__':

    # Create a pirate translator agent: translates any input into pirate speak
    pirate_agent = Agent(model=MODEL,
        name="agent_pirate_translator",
        description="This agent takes any input and translates it into pirate language, responding as a stereotypical pirate.",
        instruction=(
            "Translate every user input into pirate speak. Always respond as a classic pirate would—"
            "use phrases like 'Ahoy!', 'Avast!', 'Shiver me timbers!', and end with a hearty 'Arrr!' if possible. "
            "If the user says 'hello', reply 'Ahoy, matey!'. If they say 'how are you', reply with a pirate-themed mood. "
            "For any input, keep your answer in a fun, piratey tone."
        ),
        generate_content_config=types.GenerateContentConfig(temperature=0.7),
    )

    # Example pirate translation queries
    queries = [
        "Hello, what's your name?",
        "Can you tell me how the weather is today?",
        "How do I get to the nearest port?",
        "What is your favorite treasure?",
        "Could you translate this message into pirate language?",
    ]

    for query in queries:
        asyncio.run(send_query_to_agent(pirate_agent, query))
