from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import ConfigDict, with_config
from temporalio import activity, workflow

from pydantic_ai._run_context import RunContext
from pydantic_ai.toolsets import FunctionToolset, ToolsetTool

from ._run_context import TemporalRunContext
from ._settings import TemporalSettings


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _CallToolParams:
    name: str
    tool_args: dict[str, Any]
    serialized_run_context: Any


def temporalize_function_toolset(
    toolset: FunctionToolset,
    settings: TemporalSettings | None = None,
    tool_settings: dict[str, TemporalSettings] = {},
) -> list[Callable[..., Any]]:
    """Temporalize a function toolset.

    Args:
        toolset: The function toolset to temporalize.
        settings: The temporal settings to use.
        tool_settings: The temporal settings to use for specific tools identified by tool name.
    """
    if activities := getattr(toolset, '__temporal_activities', None):
        return activities

    id = toolset.id
    assert id is not None

    settings = settings or TemporalSettings()

    original_call_tool = toolset.call_tool
    setattr(toolset, '__original_call_tool', original_call_tool)

    @activity.defn(name=f'function_toolset__{id}__call_tool')
    async def call_tool_activity(params: _CallToolParams) -> Any:
        name = params.name
        settings_for_tool = settings.merge(tool_settings.get(name))
        ctx = TemporalRunContext.deserialize_run_context(
            params.serialized_run_context, settings_for_tool.deserialize_run_context
        )
        try:
            tool = (await toolset.get_tools(ctx))[name]
        except KeyError as e:
            raise ValueError(
                f'Tool {name!r} not found in toolset {toolset.id!r}. '
                'Removing or renaming tools during an agent run is not supported with Temporal.'
            ) from e

        return await original_call_tool(name, params.tool_args, ctx, tool)

    async def call_tool(name: str, tool_args: dict[str, Any], ctx: RunContext, tool: ToolsetTool) -> Any:
        settings_for_tool = settings.merge(tool_settings.get(name))
        serialized_run_context = TemporalRunContext.serialize_run_context(ctx, settings_for_tool.serialize_run_context)
        return await workflow.execute_activity(  # pyright: ignore[reportUnknownMemberType]
            activity=call_tool_activity,
            arg=_CallToolParams(name=name, tool_args=tool_args, serialized_run_context=serialized_run_context),
            **settings_for_tool.execute_activity_options,
        )

    toolset.call_tool = call_tool

    activities = [call_tool_activity]
    setattr(toolset, '__temporal_activities', activities)
    return activities


def untemporalize_function_toolset(toolset: FunctionToolset) -> None:
    """Untemporalize a function toolset.

    Args:
        toolset: The function toolset to untemporalize.
    """
    if not hasattr(toolset, '__temporal_activities'):
        return

    toolset.call_tool = getattr(toolset, '__original_call_tool')
    delattr(toolset, '__original_call_tool')

    delattr(toolset, '__temporal_activities')
