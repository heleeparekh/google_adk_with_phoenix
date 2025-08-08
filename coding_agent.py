import os
import time
import asyncio

from google.adk.agents import Agent, SequentialAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from phoenix.otel import register
from google.adk.tools.function_tool import FunctionTool

from dotenv import load_dotenv
load_dotenv()

# Phoenix tracing setup
tracer_provider = register(
    project_name="adk-arize-test",
    auto_instrument=True
)

MODEL = os.getenv("MODEL", "gemini-2.0-flash-001")
AGENT_APP_NAME = 'multi_agent_code_review'

session_service = InMemorySessionService()
artifact_service = InMemoryArtifactService()

# --- TOOL DEFINITION ---
def check_python_syntax(code: str) -> dict:
    """
    Checks Python code for syntax errors.
    Returns: {'status': 'ok'} or {'status': 'error', 'error': 'error message'}
    """
    import ast
    try:
        ast.parse(code)
        return {"status": "ok"}
    except SyntaxError as e:
        return {"status": "error", "error": str(e)}

syntax_checker_tool = FunctionTool(check_python_syntax)

# --- AGENT DEFINITIONS ---

# 1. Code Generator Agent
code_generator_instruction = """
You are a Code Generator agent. Given a user's request, write clear, correct, and idiomatic Python code to solve the problem.
Return only the code inside triple backticks. Do not explain, just generate code.
"""

agent_code_generator = Agent(
    model=MODEL,
    name="agent_code_generator",
    description="Generates Python code based on the user request.",
    instruction=code_generator_instruction,
    generate_content_config=types.GenerateContentConfig(temperature=0.3),
)

# 2. Code Reviewer Agent (with syntax checker tool)
code_reviewer_instruction = """
You are a Code Reviewer agent. You receive code and the original user request.
First, use the `check_python_syntax` tool with the code. If there is a syntax error, report it before anything else.
Then check the code for correctness, bugs, and style. Return a numbered list of issues (with line numbers if possible).
If the code is good, say so.
"""

agent_code_reviewer = Agent(
    model=MODEL,
    name="agent_code_reviewer",
    description="Reviews generated Python code for bugs and style issues.",
    instruction=code_reviewer_instruction,
    generate_content_config=types.GenerateContentConfig(temperature=0.3),
    tools=[syntax_checker_tool]
)

# 3. Orchestrator Agent (Sequential workflow)
class OrchestratorAgent(SequentialAgent):
    def __init__(self, name, description, code_generator, code_reviewer):
        super().__init__(
            name=name,
            description=description,
            sub_agents=[code_generator, code_reviewer]
        )

    async def call(self, *, query: str):
        # Step 1: Generate code
        code_response = await self.sub_agents[0](query=query)
        code = code_response

        # Step 2: Review code
        review_prompt = (
            f"User Request: {query}\n"
            f"Generated Code:\n{code}\n"
            "Please review this code for bugs and style issues. Start by calling the check_python_syntax tool."
        )
        review_response = await self.sub_agents[1](query=review_prompt)
        review = review_response

        # Step 3: Combine and return
        return (
            f"Generated Code:\n{code}\n\n"
            f"Code Review:\n{review}"
        )

orchestrator_agent = OrchestratorAgent(
    name="agent_orchestrator",
    description="Handles code generation requests by invoking a code generator and reviewer in sequence, and combines the results.",
    code_generator=agent_code_generator,
    code_reviewer=agent_code_reviewer
)

# --- Runner ---
async def send_query_to_agent(agent, query, user_id="user", session_id="user_session"):
    session = await session_service.create_session(app_name=AGENT_APP_NAME,
                                                   user_id=user_id,
                                                   session_id=session_id)
    print('\nUser Query: ', query)
    content = types.Content(role='user', parts=[types.Part(text=query)])

    start_time = time.time()
    runner = Runner(app_name=AGENT_APP_NAME, agent=agent,
                   artifact_service=artifact_service,
                   session_service=session_service)
    events = runner.run_async(user_id=user_id, session_id=session_id, new_message=content)
    final_response = None
    elapsed_time_ms = 0.0

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
            final_response = event.content.parts[0].text
            print(f'Agent: {event.author}')
            print(f'Response time: {elapsed_time_ms} ms\n')
            print(f'Final Response:\n{final_response}')
            print("----------------------------------------------------------\n")

    return elapsed_time_ms, final_response

# --- Main ---
if __name__ == '__main__':
    asyncio.run(send_query_to_agent(
        orchestrator_agent,
        "Write a Python function to compute the factorial of a number."
    ))

    # Try more queries!
    # queries = [
    #     "Generate a Python function that sorts a list using bubble sort.",
    # ]
    # for q in queries:
    #     asyncio.run(send_query_to_agent(orchestrator_agent, q))
