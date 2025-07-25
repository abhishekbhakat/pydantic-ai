from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Callable

import logfire  # TODO: Not always available
from opentelemetry import trace  # TODO: Not always available
from temporalio.client import ClientConfig, Plugin as ClientPlugin
from temporalio.contrib.opentelemetry import TracingInterceptor
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.runtime import OpenTelemetryConfig, Runtime, TelemetryConfig
from temporalio.service import ConnectConfig, ServiceClient
from temporalio.worker import Plugin as WorkerPlugin, WorkerConfig
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

from pydantic_ai.agent import Agent
from pydantic_ai.toolsets.abstract import AbstractToolset

from ..models import Model
from ._model import temporalize_model
from ._run_context import TemporalRunContext
from ._settings import TemporalSettings
from ._toolset import temporalize_toolset

__all__ = [
    'TemporalSettings',
    'TemporalRunContext',
    'PydanticAIPlugin',
    'LogfirePlugin',
    'AgentPlugin',
]


class PydanticAIPlugin(ClientPlugin, WorkerPlugin):
    """Temporal client and worker plugin for Pydantic AI."""

    def configure_client(self, config: ClientConfig) -> ClientConfig:
        config['data_converter'] = pydantic_data_converter
        return super().configure_client(config)

    def configure_worker(self, config: WorkerConfig) -> WorkerConfig:
        runner = config.get('workflow_runner')  # pyright: ignore[reportUnknownMemberType]
        if isinstance(runner, SandboxedWorkflowRunner):
            config['workflow_runner'] = replace(
                runner,
                restrictions=runner.restrictions.with_passthrough_modules(
                    'pydantic_ai',
                    'logfire',
                    # Imported inside `logfire._internal.json_encoder` when running `logfire.info` inside an activity with attributes to serialize
                    'attrs',
                    # Imported inside `logfire._internal.json_schema` when running `logfire.info` inside an activity with attributes to serialize
                    'numpy',
                    'pandas',
                ),
            )
        return super().configure_worker(config)


class LogfirePlugin(ClientPlugin):
    """Temporal client plugin for Logfire."""

    def configure_client(self, config: ClientConfig) -> ClientConfig:
        config['interceptors'] = [TracingInterceptor(trace.get_tracer('temporal'))]
        return super().configure_client(config)

    async def connect_service_client(self, config: ConnectConfig) -> ServiceClient:
        # TODO: Do we need this here?
        logfire.configure(console=False)
        logfire.instrument_pydantic_ai()
        logfire.instrument_httpx(capture_all=True)

        config.runtime = Runtime(telemetry=TelemetryConfig(metrics=OpenTelemetryConfig(url='http://localhost:4318')))
        return await super().connect_service_client(config)


class AgentPlugin(WorkerPlugin):
    """Temporal worker plugin for a specific Pydantic AI agent."""

    def __init__(self, agent: Agent[Any, Any], settings: TemporalSettings | None = None):
        self.activities = temporalize_agent(agent, settings)

    def configure_worker(self, config: WorkerConfig) -> WorkerConfig:
        activities: Sequence[Callable[..., Any]] = config.get('activities', [])  # pyright: ignore[reportUnknownMemberType]
        config['activities'] = [*activities, *self.activities]
        return super().configure_worker(config)


def temporalize_agent(
    agent: Agent[Any, Any],
    settings: TemporalSettings | None = None,
    temporalize_toolset_func: Callable[
        [AbstractToolset, TemporalSettings | None], list[Callable[..., Any]]
    ] = temporalize_toolset,
) -> list[Callable[..., Any]]:
    """Temporalize an agent.

    Args:
        agent: The agent to temporalize.
        settings: The temporal settings to use.
        temporalize_toolset_func: The function to use to temporalize the toolsets.
    """
    if existing_activities := getattr(agent, '__temporal_activities', None):
        return existing_activities

    settings = settings or TemporalSettings()

    # TODO: Doesn't consider model/toolsets passed at iter time, raise an error if that happens.
    # Similarly, passing event_stream_handler at iter time should raise an error.

    activities: list[Callable[..., Any]] = []
    if isinstance(agent.model, Model):
        activities.extend(temporalize_model(agent.model, settings, agent._event_stream_handler))  # pyright: ignore[reportPrivateUsage]

    def temporalize_toolset(toolset: AbstractToolset) -> None:
        activities.extend(temporalize_toolset_func(toolset, settings))

    agent.toolset.apply(temporalize_toolset)

    setattr(agent, '__temporal_activities', activities)
    return activities


# TODO: untemporalize_agent
