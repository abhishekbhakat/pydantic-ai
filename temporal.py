import asyncio
import random
from collections.abc import AsyncIterable
from datetime import timedelta

import logfire
from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.workflow import ActivityConfig
from typing_extensions import TypedDict

from pydantic_ai import Agent, RunContext
from pydantic_ai.ext.temporal import (
    AgentPlugin,
    LogfirePlugin,
    PydanticAIPlugin,
    TemporalRunContextWithDeps,
    temporalize_agent,
)
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.messages import AgentStreamEvent, HandleResponseEvent
from pydantic_ai.toolsets.function import FunctionToolset


class Deps(TypedDict):
    country: str


async def event_stream_handler(ctx: RunContext[Deps], stream: AsyncIterable[AgentStreamEvent | HandleResponseEvent]):
    logfire.info(f'{ctx.run_step=}')
    async for event in stream:
        logfire.info(f'{event=}')


toolset = FunctionToolset[Deps](id='toolset')


@toolset.tool
async def get_country(ctx: RunContext[Deps]) -> str:
    return ctx.deps['country']


@toolset.tool
def get_weather(city: str) -> str:
    return 'sunny'


agent = Agent(
    'openai:gpt-4o',
    deps_type=Deps,
    toolsets=[toolset, MCPServerStdio('python', ['-m', 'tests.mcp_server'], timeout=20, id='mcp')],
    event_stream_handler=event_stream_handler,
)

# This needs to be called in the same scope where the `agent` is bound to the workflow,
# as it modifies the `agent` object in place to swap out methods that use IO for ones that use Temporal activities.
temporalize_agent(
    agent,
    activity_config=ActivityConfig(start_to_close_timeout=timedelta(seconds=60)),
    toolset_activity_config={
        'country': ActivityConfig(start_to_close_timeout=timedelta(seconds=120)),
    },
    tool_activity_config={
        'toolset': {
            'get_country': False,
            'get_weather': ActivityConfig(start_to_close_timeout=timedelta(seconds=180)),
        },
    },
    run_context_type=TemporalRunContextWithDeps,
)

with workflow.unsafe.imports_passed_through():
    import pandas  # noqa: F401


@workflow.defn
class MyAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str, deps: Deps) -> str:
        result = await agent.run(prompt, deps=deps)
        return result.output


TASK_QUEUE = 'pydantic-ai-agent-task-queue'


def setup_logfire():
    logfire.configure(console=False)
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx(capture_all=True)


async def main():
    client = await Client.connect(
        'localhost:7233',
        plugins=[PydanticAIPlugin(), LogfirePlugin(setup_logfire)],
    )

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MyAgentWorkflow],
        plugins=[AgentPlugin(agent)],
    ):
        output = await client.execute_workflow(  # pyright: ignore[reportUnknownMemberType]
            MyAgentWorkflow.run,
            args=[
                'what is the capital of the country? what is the weather there? what is the product name?',
                Deps(country='Mexico'),
            ],
            id=f'my-agent-workflow-id-{random.random()}',
            task_queue=TASK_QUEUE,
        )
        print(output)


if __name__ == '__main__':
    asyncio.run(main())
