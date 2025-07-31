from __future__ import annotations

from typing import Any, Callable, cast

from pydantic_ai.agent import Agent
from pydantic_ai.models import Model
from pydantic_ai.toolsets.abstract import AbstractToolset
from pydantic_ai.toolsets.function import FunctionToolset

from ._model import TemporalModel
from ._settings import TemporalSettings
from ._toolset import temporalize_toolset


def temporalize_agent(
    agent: Agent[Any, Any],
    settings: TemporalSettings | None = None,
    toolset_settings: dict[str, TemporalSettings] = {},
    tool_settings: dict[str, dict[str, TemporalSettings]] = {},
    temporalize_toolset_func: Callable[
        [AbstractToolset, TemporalSettings | None, dict[str, TemporalSettings]], AbstractToolset
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
        model = TemporalModel(agent.model, settings, agent._event_stream_handler)  # pyright: ignore[reportPrivateUsage]
        activities.extend(model.activities)
        agent.model = model
    else:
        raise ValueError(
            'Model cannot be set at agent run time when using Temporal, it must be set at agent creation time.'
        )

    def temporalize_toolset(toolset: AbstractToolset) -> AbstractToolset:
        id = toolset.id
        if not id:
            raise ValueError(
                "A toolset needs to have an ID in order to be used with Temporal. The ID will be used to identify the toolset's activities within the workflow."
            )
        toolset = temporalize_toolset_func(toolset, settings.merge(toolset_settings.get(id)), tool_settings.get(id, {}))
        if hasattr(toolset, 'activities'):
            activities.extend(getattr(toolset, 'activities'))
        return toolset

    agent._function_toolset = cast(FunctionToolset, temporalize_toolset(agent._function_toolset))  # pyright: ignore[reportPrivateUsage]
    agent._user_toolsets = [temporalize_toolset(toolset) for toolset in agent._user_toolsets]  # pyright: ignore[reportPrivateUsage]

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
