import os
import asyncio
from dotenv import load_dotenv
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.llms.openai import OpenAI

load_dotenv()

class JokeEvent(Event):
    joke: str

class JokeFlow(Workflow):
    llm = OpenAI()

    @step
    async def generate_joke(self, ev: StartEvent) -> JokeEvent:
        topic = ev.topic
        prompt = f"Write your best joke about {topic}."
        response = await self.llm.acomplete(prompt)
        print(f"Joke: {response}")
        return JokeEvent(joke=str(response))

    @step
    async def critique_joke(self, ev: JokeEvent) -> StopEvent:
        joke = ev.joke
        prompt = f"Give a thorough analysis and critique of the following joke: {joke}"
        response = await self.llm.acomplete(prompt)
        return StopEvent(result=str(response))

async def run_workflow():
    w = JokeFlow(timeout=60, verbose=False)
    result = await w.run(topic="math book")
    print(str(result))

if __name__ == "__main__":
    asyncio.run(run_workflow())