import asyncio
import os
from dotenv import load_dotenv
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.llms.openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Ensure OPENAI_API_KEY is set in the environment
if "OPENAI_API_KEY" not in os.environ:
    raise EnvironmentError("OPENAI_API_KEY not found in environment variables")

class OpenAIGenerator(Workflow):
    @step()
    async def generate(self, ev: StartEvent) -> StopEvent:
        llm = OpenAI(model="gpt-3.5-turbo")  # Changed from "gpt-4o" to "gpt-4"
        response = await llm.acomplete(ev.query)
        return StopEvent(result=str(response))

async def run_workflow():
    w = OpenAIGenerator(timeout=10, verbose=False)
    result = await w.run(query="What's LlamaIndex?")
    print(result)

if __name__ == "__main__":
    asyncio.run(run_workflow())