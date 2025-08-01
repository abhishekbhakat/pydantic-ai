from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from typing import Any, Callable, Literal, overload

from temporalio import workflow
from temporalio.workflow import ActivityConfig
from typing_extensions import Never, deprecated

from pydantic_ai import (
    _utils,
    messages as _messages,
    models,
    usage as _usage,
)
from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.agent import AbstractAgent, Agent, AgentRun, RunOutputDataT, WrapperAgent
from pydantic_ai.exceptions import UserError
from pydantic_ai.ext.temporal._run_context import TemporalRunContext
from pydantic_ai.models import Model
from pydantic_ai.output import OutputDataT, OutputSpec
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import (
    Tool,
    ToolFuncEither,
)
from pydantic_ai.toolsets import AbstractToolset

from ._model import TemporalModel
from ._toolset import TemporalWrapperToolset, temporalize_toolset


class TemporalAgent(WrapperAgent[AgentDepsT, OutputDataT]):
    def __init__(
        self,
        wrapped: AbstractAgent[AgentDepsT, OutputDataT],
        activity_config: ActivityConfig = {},
        toolset_activity_config: dict[str, ActivityConfig] = {},
        tool_activity_config: dict[str, dict[str, ActivityConfig | Literal[False]]] = {},
        run_context_type: type[TemporalRunContext] = TemporalRunContext,
        temporalize_toolset_func: Callable[
            [
                AbstractToolset[Any],
                ActivityConfig,
                dict[str, ActivityConfig | Literal[False]],
                type[TemporalRunContext],
            ],
            AbstractToolset[Any],
        ] = temporalize_toolset,
    ):
        """Wrap an agent to make it compatible with Temporal.

        Args:
            wrapped: The agent to wrap.
            activity_config: The Temporal activity config to use.
            toolset_activity_config: The Temporal activity config to use for specific toolsets identified by ID.
            tool_activity_config: The Temporal activity config to use for specific tools identified by toolset ID and tool name.
            run_context_type: The type of run context to use to serialize and deserialize the run context.
            temporalize_toolset_func: The function to use to prepare the toolsets for Temporal.
        """
        super().__init__(wrapped)

        # TODO: Make this work with any AbstractAgent
        assert isinstance(wrapped, Agent)
        agent = wrapped

        activities: list[Callable[..., Any]] = []
        if not isinstance(agent.model, Model):
            raise UserError(
                'Model cannot be set at agent run time when using Temporal, it must be set at agent creation time.'
            )

        temporal_model = TemporalModel(agent.model, activity_config, agent.event_stream_handler, run_context_type)
        activities.extend(temporal_model.temporal_activities)

        def temporalize_toolset(toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
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
        temporal_toolsets = [
            temporalize_toolset(toolset)
            for toolset in [agent._function_toolset, *agent._user_toolsets]  # pyright: ignore[reportPrivateUsage]
        ]

        self._model = temporal_model
        self._toolsets = temporal_toolsets
        self._temporal_activities = activities

    @property
    def model(self) -> Model:
        return self._model

    @property
    def temporal_activities(self) -> list[Callable[..., Any]]:
        return self._temporal_activities

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, OutputDataT]]: ...

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, RunOutputDataT]]: ...

    @overload
    @deprecated('`result_type` is deprecated, use `output_type` instead.')
    def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        result_type: type[RunOutputDataT],
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, Any]]: ...

    @asynccontextmanager
    async def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AsyncIterator[AgentRun[AgentDepsT, Any]]:
        """A contextmanager which can be used to iterate over the agent graph's nodes as they are executed.

        This method builds an internal agent graph (using system prompts, tools and output schemas) and then returns an
        `AgentRun` object. The `AgentRun` can be used to async-iterate over the nodes of the graph as they are
        executed. This is the API to use if you want to consume the outputs coming from each LLM model response, or the
        stream of events coming from the execution of tools.

        The `AgentRun` also provides methods to access the full message history, new messages, and usage statistics,
        and the final result of the run once it has completed.

        For more details, see the documentation of `AgentRun`.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        async def main():
            nodes = []
            async with agent.iter('What is the capital of France?') as agent_run:
                async for node in agent_run:
                    nodes.append(node)
            print(nodes)
            '''
            [
                UserPromptNode(
                    user_prompt='What is the capital of France?',
                    instructions=None,
                    instructions_functions=[],
                    system_prompts=(),
                    system_prompt_functions=[],
                    system_prompt_dynamic_functions={},
                ),
                ModelRequestNode(
                    request=ModelRequest(
                        parts=[
                            UserPromptPart(
                                content='What is the capital of France?',
                                timestamp=datetime.datetime(...),
                            )
                        ]
                    )
                ),
                CallToolsNode(
                    model_response=ModelResponse(
                        parts=[TextPart(content='Paris')],
                        usage=Usage(
                            requests=1, request_tokens=56, response_tokens=1, total_tokens=57
                        ),
                        model_name='gpt-4o',
                        timestamp=datetime.datetime(...),
                    )
                ),
                End(data=FinalResult(output='Paris')),
            ]
            '''
            print(agent_run.result.output)
            #> Paris
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.

        Returns:
            The result of the run.
        """
        if not workflow.in_workflow():
            async with super().iter(
                user_prompt=user_prompt,
                output_type=output_type,
                message_history=message_history,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
            ) as result:
                yield result

        if model is not None:
            raise UserError(
                'Model cannot be set at agent run time when using Temporal, it must be set at agent creation time.'
            )
        if toolsets is not None:
            raise UserError(
                'Toolsets cannot be set at agent run time when using Temporal, it must be set at agent creation time.'
            )

        # We reset tools here as the temporalized function toolset is already in self._toolsets.
        with super().override(model=self._model, toolsets=self._toolsets, tools=[]):
            async with super().iter(
                user_prompt=user_prompt,
                output_type=output_type,
                message_history=message_history,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
            ) as result:
                yield result

    @contextmanager
    def override(
        self,
        *,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        model: models.Model | models.KnownModelName | str | _utils.Unset = _utils.UNSET,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | _utils.Unset = _utils.UNSET,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] | _utils.Unset = _utils.UNSET,
    ) -> Iterator[None]:
        """Context manager to temporarily override agent dependencies, model, or toolsets.

        This is particularly useful when testing.
        You can find an example of this [here](../testing.md#overriding-model-via-pytest-fixtures).

        Args:
            deps: The dependencies to use instead of the dependencies passed to the agent run.
            model: The model to use instead of the model passed to the agent run.
            toolsets: The toolsets to use instead of the toolsets passed to the agent constructor and agent run.
            tools: The tools to use instead of the tools registered with the agent.
        """
        if workflow.in_workflow():
            if _utils.is_set(model):
                raise UserError(
                    'Model cannot be contextually overridden when using Temporal, it must be set at agent creation time.'
                )
            if _utils.is_set(toolsets):
                raise UserError(
                    'Toolsets cannot be contextually overridden when using Temporal, they must be set at agent creation time.'
                )
            if _utils.is_set(tools):
                raise UserError(
                    'Tools cannot be contextually overridden when using Temporal, they must be set at agent creation time.'
                )

        with super().override(deps=deps, model=model, toolsets=toolsets, tools=tools):
            yield
