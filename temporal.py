import asyncio
import random
from collections.abc import AsyncIterable
from datetime import timedelta

import logfire
from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker
from typing_extensions import TypedDict

from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.messages import AgentStreamEvent, HandleResponseEvent
from pydantic_ai.temporal import (
    AgentPlugin,
    LogfirePlugin,
    PydanticAIPlugin,
    TemporalSettings,
)
from pydantic_ai.toolsets import FunctionToolset


class Deps(TypedDict):
    country: str


def get_country(ctx: RunContext[Deps]) -> str:
    return ctx.deps['country']


toolset = FunctionToolset[Deps](tools=[get_country], id='country')
mcp_server = MCPServerStdio(
    'python',
    ['-m', 'tests.mcp_server'],
    timeout=20,
    id='test',
)


async def event_stream_handler(
    ctx: RunContext[Deps],
    stream: AsyncIterable[AgentStreamEvent | HandleResponseEvent],
):
    logfire.info(f'{ctx.run_step=}')
    async for event in stream:
        logfire.info(f'{event=}')


agent = Agent(
    'openai:gpt-4o',
    toolsets=[toolset, mcp_server],
    event_stream_handler=event_stream_handler,
    deps_type=Deps,
)

temporal_settings = TemporalSettings(
    start_to_close_timeout=timedelta(seconds=60),
    tool_settings={  # TODO: Allow default temporal settings to be set for all activities in a toolset
        'country': {
            'get_country': TemporalSettings(start_to_close_timeout=timedelta(seconds=110)),
        },
    },
)


TASK_QUEUE = 'pydantic-ai-agent-task-queue'


@workflow.defn
class MyAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str, deps: Deps) -> str:
        result = await agent.run(prompt, deps=deps)
        return result.output


# TODO: For some reason, when I put this (specifically the temporalize_agent call) inside `async def main()`,
# we get tons of errors.
plugin = AgentPlugin(agent, temporal_settings)


async def main():
    client = await Client.connect(
        'localhost:7233',
        plugins=[PydanticAIPlugin(), LogfirePlugin()],
    )

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MyAgentWorkflow],
        plugins=[plugin],
    ):
        output = await client.execute_workflow(  # pyright: ignore[reportUnknownMemberType]
            MyAgentWorkflow.run,
            args=[
                'what is the capital of the capital of the country? and what is the product name?',
                Deps(country='Mexico'),
            ],
            id=f'my-agent-workflow-id-{random.random()}',
            task_queue=TASK_QUEUE,
        )
        print(output)


if __name__ == '__main__':
    asyncio.run(main())
