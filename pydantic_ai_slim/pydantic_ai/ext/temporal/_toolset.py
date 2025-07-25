from __future__ import annotations

from typing import Any, Callable

from pydantic_ai.mcp import MCPServer
from pydantic_ai.toolsets.abstract import AbstractToolset
from pydantic_ai.toolsets.function import FunctionToolset

from ._function_toolset import temporalize_function_toolset, untemporalize_function_toolset
from ._mcp_server import temporalize_mcp_server, untemporalize_mcp_server
from ._settings import TemporalSettings


def temporalize_toolset(
    toolset: AbstractToolset, settings: TemporalSettings | None, tool_settings: dict[str, TemporalSettings] = {}
) -> list[Callable[..., Any]]:
    """Temporalize a toolset.

    Args:
        toolset: The toolset to temporalize.
        settings: The temporal settings to use.
        tool_settings: The temporal settings to use for specific tools identified by tool name.
    """
    if isinstance(toolset, FunctionToolset):
        return temporalize_function_toolset(toolset, settings, tool_settings)
    elif isinstance(toolset, MCPServer):
        return temporalize_mcp_server(toolset, settings, tool_settings)
    else:
        return []


def untemporalize_toolset(toolset: AbstractToolset) -> None:
    """Untemporalize a toolset.

    Args:
        toolset: The toolset to untemporalize.
    """
    if isinstance(toolset, FunctionToolset):
        untemporalize_function_toolset(toolset)
    elif isinstance(toolset, MCPServer):
        untemporalize_mcp_server(toolset)
