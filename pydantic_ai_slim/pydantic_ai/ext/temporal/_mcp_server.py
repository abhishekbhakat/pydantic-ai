from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import ConfigDict, with_config
from temporalio import activity, workflow

from pydantic_ai._run_context import RunContext
from pydantic_ai.mcp import MCPServer, ToolResult
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets.abstract import ToolsetTool

from ._run_context import TemporalRunContext
from ._settings import TemporalSettings


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _GetToolsParams:
    serialized_run_context: Any


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _CallToolParams:
    name: str
    tool_args: dict[str, Any]
    serialized_run_context: Any
    tool_def: ToolDefinition


def temporalize_mcp_server(
    server: MCPServer,
    settings: TemporalSettings | None = None,
    tool_settings: dict[str, TemporalSettings] = {},
) -> list[Callable[..., Any]]:
    """Temporalize an MCP server.

    Args:
        server: The MCP server to temporalize.
        settings: The temporal settings to use.
        tool_settings: The temporal settings to use for each tool.
    """
    if activities := getattr(server, '__temporal_activities', None):
        return activities

    id = server.id
    assert id is not None

    settings = settings or TemporalSettings()

    original_get_tools = server.get_tools
    original_call_tool = server.call_tool
    setattr(server, '__original_get_tools', original_get_tools)
    setattr(server, '__original_call_tool', original_call_tool)

    @activity.defn(name=f'mcp_server__{id}__get_tools')
    async def get_tools_activity(params: _GetToolsParams) -> dict[str, ToolDefinition]:
        run_context = TemporalRunContext.deserialize_run_context(
            params.serialized_run_context, settings.deserialize_run_context
        )
        return {name: tool.tool_def for name, tool in (await original_get_tools(run_context)).items()}

    @activity.defn(name=f'mcp_server__{id}__call_tool')
    async def call_tool_activity(params: _CallToolParams) -> ToolResult:
        run_context = TemporalRunContext.deserialize_run_context(
            params.serialized_run_context, settings.deserialize_run_context
        )
        return await original_call_tool(
            params.name,
            params.tool_args,
            run_context,
            server._toolset_tool_for_tool_def(params.tool_def),  # pyright: ignore[reportPrivateUsage]
        )

    async def get_tools(ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        serialized_run_context = TemporalRunContext.serialize_run_context(ctx, settings.serialize_run_context)
        tool_defs = await workflow.execute_activity(  # pyright: ignore[reportUnknownMemberType]
            activity=get_tools_activity,
            arg=_GetToolsParams(serialized_run_context=serialized_run_context),
            **settings.execute_activity_options,
        )
        return {
            name: server._toolset_tool_for_tool_def(tool_def)  # pyright: ignore[reportPrivateUsage]
            for name, tool_def in tool_defs.items()
        }

    async def call_tool(
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
    ) -> ToolResult:
        serialized_run_context = TemporalRunContext.serialize_run_context(ctx, settings.serialize_run_context)
        return await workflow.execute_activity(  # pyright: ignore[reportUnknownMemberType]
            activity=call_tool_activity,
            arg=_CallToolParams(
                name=name,
                tool_args=tool_args,
                serialized_run_context=serialized_run_context,
                tool_def=tool.tool_def,
            ),
            **tool_settings.get(name, settings).execute_activity_options,
        )

    server.get_tools = get_tools
    server.call_tool = call_tool

    activities = [get_tools_activity, call_tool_activity]
    setattr(server, '__temporal_activities', activities)
    return activities


def untemporalize_mcp_server(server: MCPServer) -> None:
    """Untemporalize an MCP server.

    Args:
        server: The MCP server to untemporalize.
    """
    if not hasattr(server, '__temporal_activities'):
        return

    server.get_tools = getattr(server, '__original_get_tools')
    server.call_tool = getattr(server, '__original_call_tool')
    delattr(server, '__original_get_tools')
    delattr(server, '__original_call_tool')

    delattr(server, '__temporal_activities')
