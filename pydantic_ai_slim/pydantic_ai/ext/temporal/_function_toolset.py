from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, cast

from pydantic import ConfigDict, with_config
from temporalio import activity, workflow

from pydantic_ai._run_context import RunContext
from pydantic_ai.toolsets import FunctionToolset, ToolsetTool
from pydantic_ai.toolsets.wrapper import WrapperToolset

from ._run_context import TemporalRunContext
from ._settings import TemporalSettings


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _CallToolParams:
    name: str
    tool_args: dict[str, Any]
    serialized_run_context: Any


class TemporalFunctionToolset(WrapperToolset[Any]):
    def __init__(
        self,
        toolset: FunctionToolset,
        settings: TemporalSettings | None = None,
        tool_settings: dict[str, TemporalSettings] = {},
    ):
        super().__init__(toolset)
        self.settings = settings or TemporalSettings()
        self.tool_settings = tool_settings

        id = toolset.id
        assert id is not None

        @activity.defn(name=f'function_toolset__{id}__call_tool')
        async def call_tool_activity(params: _CallToolParams) -> Any:
            name = params.name
            settings_for_tool = self.settings.merge(self.tool_settings.get(name))
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

            return await self.wrapped.call_tool(name, params.tool_args, ctx, tool)

        self.call_tool_activity = call_tool_activity

    @property
    def wrapped_function_toolset(self) -> FunctionToolset:
        return cast(FunctionToolset, self.wrapped)

    @property
    def activities(self) -> list[Callable[..., Any]]:
        return [self.call_tool_activity]

    def tool(self, *args: Any, **kwargs: Any) -> Any:
        return self.wrapped_function_toolset.tool(*args, **kwargs)

    def add_function(self, *args: Any, **kwargs: Any) -> None:
        return self.wrapped_function_toolset.add_function(*args, **kwargs)

    def add_tool(self, *args: Any, **kwargs: Any) -> None:
        return self.wrapped_function_toolset.add_tool(*args, **kwargs)

    async def call_tool(self, name: str, tool_args: dict[str, Any], ctx: RunContext, tool: ToolsetTool) -> Any:
        settings_for_tool = self.settings.merge(self.tool_settings.get(name))
        serialized_run_context = TemporalRunContext.serialize_run_context(ctx, settings_for_tool.serialize_run_context)
        return await workflow.execute_activity(  # pyright: ignore[reportUnknownMemberType]
            activity=self.call_tool_activity,
            arg=_CallToolParams(name=name, tool_args=tool_args, serialized_run_context=serialized_run_context),
            **settings_for_tool.execute_activity_options,
        )
