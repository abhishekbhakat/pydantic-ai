from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Callable

from temporalio.client import ClientConfig, Plugin as ClientPlugin
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Plugin as WorkerPlugin, WorkerConfig
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

from pydantic_ai.agent import Agent

from ._agent import temporalize_agent
from ._logfire import LogfirePlugin
from ._run_context import TemporalRunContext, TemporalRunContextWithDeps

__all__ = [
    'TemporalRunContext',
    'TemporalRunContextWithDeps',
    'PydanticAIPlugin',
    'LogfirePlugin',
    'AgentPlugin',
    'temporalize_agent',
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


class AgentPlugin(WorkerPlugin):
    """Temporal worker plugin for a specific Pydantic AI agent."""

    def __init__(self, agent: Agent[Any, Any]):
        self.agent = agent

    def configure_worker(self, config: WorkerConfig) -> WorkerConfig:
        agent_activities = getattr(self.agent, '__temporal_activities', None)
        if agent_activities is None:
            raise ValueError('The agent has not been prepared for Temporal yet, call `temporalize_agent(agent)` first.')

        activities: Sequence[Callable[..., Any]] = config.get('activities', [])  # pyright: ignore[reportUnknownMemberType]
        config['activities'] = [*activities, *agent_activities]
        return super().configure_worker(config)
