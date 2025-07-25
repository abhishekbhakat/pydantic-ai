from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Callable

from pydantic_ai.agent import Agent
from pydantic_ai.models import Model
from pydantic_ai.toolsets.abstract import AbstractToolset

from ._model import temporalize_model, untemporalize_model
from ._settings import TemporalSettings
from ._toolset import temporalize_toolset, untemporalize_toolset


def temporalize_agent(
    agent: Agent[Any, Any],
    settings: TemporalSettings | None = None,
    toolset_settings: dict[str, TemporalSettings] = {},
    tool_settings: dict[str, dict[str, TemporalSettings]] = {},
    temporalize_toolset_func: Callable[
        [AbstractToolset, TemporalSettings | None, dict[str, TemporalSettings]], list[Callable[..., Any]]
    ] = temporalize_toolset,
) -> list[Callable[..., Any]]:
    """Temporalize an agent.

    Args:
        agent: The agent to temporalize.
        settings: The temporal settings to use.
        toolset_settings: The temporal settings to use for specific toolsets identified by ID.
        tool_settings: The temporal settings to use for specific tools identified by toolset ID and tool name.
        temporalize_toolset_func: The function to use to temporalize the toolsets.
    """
    if existing_activities := getattr(agent, '__temporal_activities', None):
        return existing_activities

    settings = settings or TemporalSettings()

    activities: list[Callable[..., Any]] = []
    if isinstance(agent.model, Model):
        activities.extend(temporalize_model(agent.model, settings, agent._event_stream_handler))  # pyright: ignore[reportPrivateUsage]

    def temporalize_toolset(toolset: AbstractToolset) -> None:
        id = toolset.id
        if not id:
            raise ValueError(
                "A toolset needs to have an ID in order to be used with Temporal. The ID will be used to identify the toolset's activities within the workflow."
            )
        activities.extend(
            temporalize_toolset_func(toolset, settings.merge(toolset_settings.get(id)), tool_settings.get(id, {}))
        )

    agent.toolset.apply(temporalize_toolset)

    original_iter = agent.iter
    original_override = agent.override
    setattr(agent, '__original_iter', original_iter)
    setattr(agent, '__original_override', original_override)

    def iter(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get('model') is not None:
            raise ValueError(
                'Model cannot be set at agent run time when using Temporal, it must be set at agent creation time.'
            )
        if kwargs.get('toolsets') is not None:
            raise ValueError(
                'Toolsets cannot be set at agent run time when using Temporal, it must be set at agent creation time.'
            )
        if kwargs.get('event_stream_handler') is not None:
            raise ValueError(
                'Event stream handler cannot be set at agent run time when using Temporal, it must be set at agent creation time.'
            )

        return original_iter(*args, **kwargs)

    def override(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get('model') is not None:
            raise ValueError('Model cannot be overridden when using Temporal, it must be set at agent creation time.')
        if kwargs.get('toolsets') is not None:
            raise ValueError(
                'Toolsets cannot be overridden when using Temporal, it must be set at agent creation time.'
            )
        return original_override(*args, **kwargs)

    agent.iter = iter
    agent.override = override

    setattr(agent, '__temporal_activities', activities)
    return activities


def untemporalize_agent(agent: Agent[Any, Any]) -> None:
    """Untemporalize an agent.

    Args:
        agent: The agent to untemporalize.
    """
    if not hasattr(agent, '__temporal_activities'):
        return

    if isinstance(agent.model, Model):
        untemporalize_model(agent.model)

    agent.toolset.apply(untemporalize_toolset)

    agent.iter = getattr(agent, '__original_iter')
    agent.override = getattr(agent, '__original_override')
    delattr(agent, '__original_iter')
    delattr(agent, '__original_override')

    delattr(agent, '__temporal_activities')


@contextmanager
def temporalized_agent(agent: Agent[Any, Any], settings: TemporalSettings | None = None) -> Generator[None, None, None]:
    temporalize_agent(agent, settings)
    try:
        yield
    finally:
        untemporalize_agent(agent)
