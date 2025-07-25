from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from mcp import types as mcp_types
from pydantic import ConfigDict, with_config
from temporalio import activity, workflow

from pydantic_ai.mcp import MCPServer, ToolResult

from ._settings import TemporalSettings


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _CallToolParams:
    name: str
    tool_args: dict[str, Any]
    metadata: dict[str, Any] | None = None


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

    original_list_tools = server.list_tools
    original_direct_call_tool = server.direct_call_tool
    setattr(server, '__original_list_tools', original_list_tools)
    setattr(server, '__original_direct_call_tool', original_direct_call_tool)

    @activity.defn(name=f'mcp_server__{id}__list_tools')
    async def list_tools_activity() -> list[mcp_types.Tool]:
        return await original_list_tools()

    @activity.defn(name=f'mcp_server__{id}__call_tool')
    async def call_tool_activity(params: _CallToolParams) -> ToolResult:
        return await original_direct_call_tool(params.name, params.tool_args, params.metadata)

    async def list_tools() -> list[mcp_types.Tool]:
        return await workflow.execute_activity(  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]
            activity=list_tools_activity,
            **settings.execute_activity_options,
        )

    async def direct_call_tool(
        name: str,
        args: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        return await workflow.execute_activity(  # pyright: ignore[reportUnknownMemberType]
            activity=call_tool_activity,
            arg=_CallToolParams(name=name, tool_args=args, metadata=metadata),
            **tool_settings.get(name, settings).execute_activity_options,
        )

    server.list_tools = list_tools
    server.direct_call_tool = direct_call_tool

    activities = [list_tools_activity, call_tool_activity]
    setattr(server, '__temporal_activities', activities)
    return activities


def untemporalize_mcp_server(server: MCPServer) -> None:
    """Untemporalize an MCP server.

    Args:
        server: The MCP server to untemporalize.
    """
    if not hasattr(server, '__temporal_activities'):
        return

    server.list_tools = getattr(server, '__original_list_tools')
    server.direct_call_tool = getattr(server, '__original_direct_call_tool')
    delattr(server, '__original_list_tools')
    delattr(server, '__original_direct_call_tool')

    delattr(server, '__temporal_activities')
