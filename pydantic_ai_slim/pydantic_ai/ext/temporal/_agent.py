from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Callable, Literal

from temporalio import workflow
from temporalio.workflow import ActivityConfig

from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.ext.temporal._run_context import TemporalRunContext
from pydantic_ai.models import Model
from pydantic_ai.toolsets.abstract import AbstractToolset

from ._model import TemporalModel
from ._toolset import TemporalWrapperToolset, temporalize_toolset


def temporalize_agent(  # noqa: C901
    agent: Agent[Any, Any],
    activity_config: ActivityConfig = {},
    toolset_activity_config: dict[str, ActivityConfig] = {},
    tool_activity_config: dict[str, dict[str, ActivityConfig | Literal[False]]] = {},
    run_context_type: type[TemporalRunContext] = TemporalRunContext,
    temporalize_toolset_func: Callable[
        [AbstractToolset, ActivityConfig, dict[str, ActivityConfig | Literal[False]], type[TemporalRunContext]],
        AbstractToolset,
    ] = temporalize_toolset,
) -> Agent[Any, Any]:
    """Temporalize an agent.

    Args:
        agent: The agent to temporalize.
        activity_config: The Temporal activity config to use.
        toolset_activity_config: The Temporal activity config to use for specific toolsets identified by ID.
        tool_activity_config: The Temporal activity config to use for specific tools identified by toolset ID and tool name.
        run_context_type: The type of run context to use to serialize and deserialize the run context.
        temporalize_toolset_func: The function to use to prepare the toolsets for Temporal.
    """
    if getattr(agent, '__temporal_activities', None):
        return agent

    activities: list[Callable[..., Any]] = []
    if not isinstance(agent.model, Model):
        raise UserError(
            'Model cannot be set at agent run time when using Temporal, it must be set at agent creation time.'
        )

    temporal_model = TemporalModel(agent.model, activity_config, agent.event_stream_handler, run_context_type)
    activities.extend(temporal_model.temporal_activities)

    def temporalize_toolset(toolset: AbstractToolset) -> AbstractToolset:
        id = toolset.id
        if not id:
            raise UserError(
                "Toolsets that implement their own tool calling need to have an ID in order to be used with Temporal. The ID will be used to identify the toolset's activities within the workflow."
            )
        toolset = temporalize_toolset_func(
            toolset,
            activity_config | toolset_activity_config.get(id, {}),
            tool_activity_config.get(id, {}),
            run_context_type,
        )
        if isinstance(toolset, TemporalWrapperToolset):
            activities.extend(toolset.temporal_activities)
        return toolset

    # TODO: Use public methods so others can replicate this
    temporal_toolsets = [temporalize_toolset(toolset) for toolset in [agent._function_toolset, *agent._user_toolsets]]  # pyright: ignore[reportPrivateUsage]

    original_iter = agent.iter
    original_override = agent.override

    def iter(*args: Any, **kwargs: Any) -> Any:
        if not workflow.in_workflow():
            return original_iter(*args, **kwargs)

        if kwargs.get('model') is not None:
            raise UserError(
                'Model cannot be set at agent run time when using Temporal, it must be set at agent creation time.'
            )
        if kwargs.get('toolsets') is not None:
            raise UserError(
                'Toolsets cannot be set at agent run time when using Temporal, it must be set at agent creation time.'
            )
        if kwargs.get('event_stream_handler') is not None:
            # TODO: iter won't have event_stream_handler, run/_sync/_stream will
            raise UserError(
                'Event stream handler cannot be set at agent run time when using Temporal, it must be set at agent creation time.'
            )

        @asynccontextmanager
        async def async_iter():
            # We reset tools here as the temporalized function toolset is already in temporal_toolsets.
            with agent.override(model=temporal_model, toolsets=temporal_toolsets, tools=[]):
                async with original_iter(*args, **kwargs) as result:
                    yield result

        return async_iter()

    def override(*args: Any, **kwargs: Any) -> Any:
        if not workflow.in_workflow():
            return original_override(*args, **kwargs)

        if kwargs.get('model') not in (None, temporal_model):
            raise UserError(
                'Model cannot be contextually overridden when using Temporal, it must be set at agent creation time.'
            )
        if kwargs.get('toolsets') not in (None, temporal_toolsets):
            raise UserError(
                'Toolsets cannot be contextually overridden when using Temporal, they must be set at agent creation time.'
            )
        if kwargs.get('tools') not in (None, []):
            raise UserError(
                'Tools cannot be contextually overridden when using Temporal, they must be set at agent creation time.'
            )
        return original_override(*args, **kwargs)

    def tool(*args: Any, **kwargs: Any) -> Any:
        raise UserError('New tools cannot be registered after an agent has been prepared for Temporal.')

    agent.iter = iter
    agent.override = override
    agent.tool = tool
    agent.tool_plain = tool

    setattr(agent, '__temporal_activities', activities)
    return agent
