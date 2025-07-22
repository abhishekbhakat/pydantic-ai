# /// script
# dependencies = [
#   "temporalio",
#   "logfire",
# ]
# ///
import asyncio
import random
from collections.abc import AsyncIterable
from datetime import timedelta

from temporalio import workflow
from temporalio.client import Client
from temporalio.contrib.opentelemetry import TracingInterceptor
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.runtime import OpenTelemetryConfig, Runtime, TelemetryConfig
from temporalio.worker import Worker
from typing_extensions import TypedDict

with workflow.unsafe.imports_passed_through():
    from pydantic_ai import Agent
    from pydantic_ai._run_context import RunContext
    from pydantic_ai.mcp import MCPServerStdio
    from pydantic_ai.messages import AgentStreamEvent, HandleResponseEvent
    from pydantic_ai.temporal import (
        TemporalSettings,
        initialize_temporal,
    )
    from pydantic_ai.temporal.agent import temporalize_agent
    from pydantic_ai.toolsets.function import FunctionToolset

    initialize_temporal()

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
        print(f'{ctx.run_step=}')
        async for event in stream:
            print(event)

    my_agent = Agent(
        'openai:gpt-4o',
        toolsets=[toolset, mcp_server],
        event_stream_handler=event_stream_handler,
        deps_type=Deps,
    )

    temporal_settings = TemporalSettings(
        start_to_close_timeout=timedelta(seconds=60),
        tool_settings={  # TODO: Allow default temporal settings to be set for an entire toolset
            'country': {
                'get_country': TemporalSettings(start_to_close_timeout=timedelta(seconds=110)),
            },
        },
    )
    activities = temporalize_agent(my_agent, temporal_settings)


def init_runtime_with_telemetry() -> Runtime:
    # import logfire

    # logfire.configure(send_to_logfire=True, service_version='0.0.1', console=False)
    # logfire.instrument_pydantic_ai()
    # logfire.instrument_httpx(capture_all=True)

    # Setup SDK metrics to OTel endpoint
    return Runtime(telemetry=TelemetryConfig(metrics=OpenTelemetryConfig(url='http://localhost:4318')))


# Basic workflow that logs and invokes an activity
@workflow.defn
class MyAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str, deps: Deps) -> str:
        result = await my_agent.run(prompt, deps=deps)
        return result.output


async def main():
    client = await Client.connect(
        'localhost:7233',
        interceptors=[TracingInterceptor()],
        data_converter=pydantic_data_converter,
        runtime=init_runtime_with_telemetry(),
    )

    async with Worker(
        client,
        task_queue='my-agent-task-queue',
        workflows=[MyAgentWorkflow],
        activities=activities,
    ):
        output = await client.execute_workflow(  # pyright: ignore[reportUnknownMemberType]
            MyAgentWorkflow.run,
            args=[
                'what is the capital of the capital of the country? and what is the product name?',
                Deps(country='Mexico'),
            ],
            id=f'my-agent-workflow-id-{random.random()}',
            task_queue='my-agent-task-queue',
        )
        print(output)


if __name__ == '__main__':
    asyncio.run(main())
