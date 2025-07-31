from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import ConfigDict, with_config
from temporalio import activity, workflow

from pydantic_ai._run_context import RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPServer, ToolResult
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai.toolsets.wrapper import WrapperToolset

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


class TemporalMCPServer(WrapperToolset[Any]):
    def __init__(
        self,
        server: MCPServer,
        settings: TemporalSettings | None = None,
        tool_settings: dict[str, TemporalSettings | Literal[False]] = {},
    ):
        super().__init__(server)
        self.settings = settings or TemporalSettings()
        self.tool_settings = tool_settings

        id = server.id
        assert id is not None

        @activity.defn(name=f'mcp_server__{id}__get_tools')
        async def get_tools_activity(params: _GetToolsParams) -> dict[str, ToolDefinition]:
            run_context = TemporalRunContext.deserialize_run_context(
                params.serialized_run_context, self.settings.deserialize_run_context
            )
            return {name: tool.tool_def for name, tool in (await self.wrapped.get_tools(run_context)).items()}

        self.get_tools_activity = get_tools_activity

        @activity.defn(name=f'mcp_server__{id}__call_tool')
        async def call_tool_activity(params: _CallToolParams) -> ToolResult:
            run_context = TemporalRunContext.deserialize_run_context(
                params.serialized_run_context, self.settings.deserialize_run_context
            )
            return await self.wrapped.call_tool(
                params.name,
                params.tool_args,
                run_context,
                self.wrapped_server._toolset_tool_for_tool_def(params.tool_def),  # pyright: ignore[reportPrivateUsage]
            )

        self.call_tool_activity = call_tool_activity

    @property
    def wrapped_server(self) -> MCPServer:
        assert isinstance(self.wrapped, MCPServer)
        return self.wrapped

    @property
    def activities(self) -> list[Callable[..., Any]]:
        return [self.get_tools_activity, self.call_tool_activity]

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        serialized_run_context = TemporalRunContext.serialize_run_context(ctx, self.settings.serialize_run_context)
        tool_defs = await workflow.execute_activity(  # pyright: ignore[reportUnknownMemberType]
            activity=self.get_tools_activity,
            arg=_GetToolsParams(serialized_run_context=serialized_run_context),
            **self.settings.execute_activity_options,
        )
        return {
            name: self.wrapped_server._toolset_tool_for_tool_def(tool_def)  # pyright: ignore[reportPrivateUsage]
            for name, tool_def in tool_defs.items()
        }

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
    ) -> ToolResult:
        settings_for_tool = self.tool_settings.get(name)
        if settings_for_tool is False:
            raise UserError('Disabling running an MCP tool in a Temporal activity is not possible.')

        settings_for_tool = self.settings.merge(settings_for_tool)
        serialized_run_context = TemporalRunContext.serialize_run_context(ctx, settings_for_tool.serialize_run_context)
        return await workflow.execute_activity(  # pyright: ignore[reportUnknownMemberType]
            activity=self.call_tool_activity,
            arg=_CallToolParams(
                name=name,
                tool_args=tool_args,
                serialized_run_context=serialized_run_context,
                tool_def=tool.tool_def,
            ),
            **settings_for_tool.execute_activity_options,
        )
