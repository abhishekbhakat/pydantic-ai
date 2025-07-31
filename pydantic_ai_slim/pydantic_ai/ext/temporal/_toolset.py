from __future__ import annotations

from pydantic_ai.mcp import MCPServer
from pydantic_ai.toolsets.abstract import AbstractToolset
from pydantic_ai.toolsets.function import FunctionToolset

from ._function_toolset import TemporalFunctionToolset
from ._mcp_server import TemporalMCPServer
from ._settings import TemporalSettings


def temporalize_toolset(
    toolset: AbstractToolset, settings: TemporalSettings | None, tool_settings: dict[str, TemporalSettings] = {}
) -> AbstractToolset:
    """Temporalize a toolset.

    Args:
        toolset: The toolset to temporalize.
        settings: The temporal settings to use.
        tool_settings: The temporal settings to use for specific tools identified by tool name.
    """
    if isinstance(toolset, FunctionToolset):
        return TemporalFunctionToolset(toolset, settings, tool_settings)
    elif isinstance(toolset, MCPServer):
        return TemporalMCPServer(toolset, settings, tool_settings)
    else:
        return toolset
