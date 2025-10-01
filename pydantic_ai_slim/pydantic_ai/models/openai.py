from __future__ import annotations as _annotations

import base64
import warnings
from collections.abc import AsyncIterable, AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import partialmethod
from typing import Any, Literal, cast, overload

from pydantic_core import to_json
from typing_extensions import assert_never, deprecated

from .. import UnexpectedModelBehavior, _utils
from .._output import OutputObjectDefinition
from .._run_context import RunContext
from .._utils import guard_tool_call_id as _guard_tool_call_id, number_to_datetime
from ..messages import (
    AudioUrl,
    BinaryContent,
    DocumentUrl,
    FinishReason,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponseStreamEvent,
    RetryPromptPart,
    SystemPromptPart,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from ..profiles import ModelProfileSpec
from ..profiles.openai import OpenAIModelProfile, OpenAISystemPromptRole
from ..providers import Provider, infer_provider
from ..settings import ModelSettings
from ..tools import ToolDefinition
from . import Model, ModelRequestParameters, StreamedResponse, check_allow_model_requests, download_item
from ._openai_compat import (
    OpenAICompatStreamedResponse,
    combine_tool_call_ids,
    completions_create as _compat_completions_create,
    get_responses_builtin_tools,
    get_responses_previous_response_id_and_new_messages,
    get_responses_reasoning,
    get_responses_tools,
    map_code_interpreter_tool_call,
    map_responses_json_schema,
    map_responses_messages,
    map_responses_tool_definition,
    map_responses_user_prompt,
    map_tool_definition,
    map_usage,
    map_web_search_tool_call,
    process_response,
    process_responses_response,
    process_streamed_response,
    responses_create,
)

try:
    from openai import AsyncOpenAI, AsyncStream, NotGiven
    from openai.types import AllModels, chat, responses
    from openai.types.chat import (
        ChatCompletionChunk,
        ChatCompletionContentPartImageParam,
        ChatCompletionContentPartInputAudioParam,
        ChatCompletionContentPartParam,
        ChatCompletionContentPartTextParam,
    )
    from openai.types.chat.chat_completion_content_part_image_param import ImageURL
    from openai.types.chat.chat_completion_content_part_input_audio_param import InputAudio
    from openai.types.chat.chat_completion_content_part_param import File, FileFile
    from openai.types.chat.chat_completion_prediction_content_param import ChatCompletionPredictionContentParam
    from openai.types.responses import ComputerToolParam, FileSearchToolParam, WebSearchToolParam
    from openai.types.responses.response_status import ResponseStatus
    from openai.types.shared import ReasoningEffort
    from openai.types.shared_params import Reasoning
except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` to use the OpenAI model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

OpenAIStreamedResponse = OpenAICompatStreamedResponse

__all__ = (
    'OpenAIModel',
    'OpenAIChatModel',
    'OpenAIResponsesModel',
    'OpenAIModelSettings',
    'OpenAIChatModelSettings',
    'OpenAIResponsesModelSettings',
    'OpenAIModelName',
)

OpenAIModelName = str | AllModels
"""
Possible OpenAI model names.

Since OpenAI supports a variety of date-stamped models, we explicitly list the latest models but
allow any name in the type hints.
See [the OpenAI docs](https://platform.openai.com/docs/models) for a full list.

Using this more broad type for the model name instead of the ChatModel definition
allows this model to be used more easily with other model types (ie, Ollama, Deepseek).
"""


_CHAT_FINISH_REASON_MAP: dict[
    Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call'], FinishReason
] = {
    'stop': 'stop',
    'length': 'length',
    'tool_calls': 'tool_call',
    'content_filter': 'content_filter',
    'function_call': 'tool_call',
}

_RESPONSES_FINISH_REASON_MAP: dict[Literal['max_output_tokens', 'content_filter'] | ResponseStatus, FinishReason] = {
    'max_output_tokens': 'length',
    'content_filter': 'content_filter',
    'completed': 'stop',
    'cancelled': 'error',
    'failed': 'error',
}


class OpenAIChatModelSettings(ModelSettings, total=False):
    """Settings used for an OpenAI model request."""

    # ALL FIELDS MUST BE `openai_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    openai_reasoning_effort: ReasoningEffort
    """Constrains effort on reasoning for [reasoning models](https://platform.openai.com/docs/guides/reasoning).

    Currently supported values are `low`, `medium`, and `high`. Reducing reasoning effort can
    result in faster responses and fewer tokens used on reasoning in a response.
    """

    openai_logprobs: bool
    """Include log probabilities in the response."""

    openai_top_logprobs: int
    """Include log probabilities of the top n tokens in the response."""

    openai_user: str
    """A unique identifier representing the end-user, which can help OpenAI monitor and detect abuse.

    See [OpenAI's safety best practices](https://platform.openai.com/docs/guides/safety-best-practices#end-user-ids) for more details.
    """

    openai_service_tier: Literal['auto', 'default', 'flex', 'priority']
    """The service tier to use for the model request.

    Currently supported values are `auto`, `default`, `flex`, and `priority`.
    For more information, see [OpenAI's service tiers documentation](https://platform.openai.com/docs/api-reference/chat/object#chat/object-service_tier).
    """

    openai_prediction: ChatCompletionPredictionContentParam
    """Enables [predictive outputs](https://platform.openai.com/docs/guides/predicted-outputs).

    This feature is currently only supported for some OpenAI models.
    """


@deprecated('Use `OpenAIChatModelSettings` instead.')
class OpenAIModelSettings(OpenAIChatModelSettings, total=False):
    """Deprecated alias for `OpenAIChatModelSettings`."""


class OpenAIResponsesModelSettings(OpenAIChatModelSettings, total=False):
    """Settings used for an OpenAI Responses model request.

    ALL FIELDS MUST BE `openai_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    openai_builtin_tools: Sequence[FileSearchToolParam | WebSearchToolParam | ComputerToolParam]
    """The provided OpenAI built-in tools to use.

    See [OpenAI's built-in tools](https://platform.openai.com/docs/guides/tools?api-mode=responses) for more details.
    """

    openai_reasoning_generate_summary: Literal['detailed', 'concise']
    """Deprecated alias for `openai_reasoning_summary`."""

    openai_reasoning_summary: Literal['detailed', 'concise']
    """A summary of the reasoning performed by the model.

    This can be useful for debugging and understanding the model's reasoning process.
    One of `concise` or `detailed`.

    Check the [OpenAI Reasoning documentation](https://platform.openai.com/docs/guides/reasoning?api-mode=responses#reasoning-summaries)
    for more details.
    """

    openai_send_reasoning_ids: bool
    """Whether to send the unique IDs of reasoning, text, and function call parts from the message history to the model. Enabled by default for reasoning models.

    This can result in errors like `"Item 'rs_123' of type 'reasoning' was provided without its required following item."`
    if the message history you're sending does not match exactly what was received from the Responses API in a previous response,
    for example if you're using a [history processor](../../message-history.md#processing-message-history).
    In that case, you'll want to disable this.
    """

    openai_truncation: Literal['disabled', 'auto']
    """The truncation strategy to use for the model response.

    It can be either:
    - `disabled` (default): If a model response will exceed the context window size for a model, the
        request will fail with a 400 error.
    - `auto`: If the context of this response and previous ones exceeds the model's context window size,
        the model will truncate the response to fit the context window by dropping input items in the
        middle of the conversation.
    """

    openai_text_verbosity: Literal['low', 'medium', 'high']
    """Constrains the verbosity of the model's text response.

    Lower values will result in more concise responses, while higher values will
    result in more verbose responses. Currently supported values are `low`,
    `medium`, and `high`.
    """

    openai_previous_response_id: Literal['auto'] | str
    """The ID of a previous response from the model to use as the starting point for a continued conversation.

    When set to `'auto'`, the request automatically uses the most recent
    `provider_response_id` from the message history and omits earlier messages.

    This enables the model to use server-side conversation state and faithfully reference previous reasoning.
    See the [OpenAI Responses API documentation](https://platform.openai.com/docs/guides/reasoning#keeping-reasoning-items-in-context)
    for more information.
    """

    openai_include_code_execution_outputs: bool
    """Whether to include the code execution results in the response.

    Corresponds to the `code_interpreter_call.outputs` value of the `include` parameter in the Responses API.
    """

    openai_include_web_search_sources: bool
    """Whether to include the web search results in the response.

    Corresponds to the `web_search_call.action.sources` value of the `include` parameter in the Responses API.
    """


@dataclass(init=False)
class OpenAIChatModel(Model):
    """A model that uses the OpenAI API.

    Internally, this uses the [OpenAI Python client](https://github.com/openai/openai-python) to interact with the API.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    client: AsyncOpenAI = field(repr=False)

    _model_name: OpenAIModelName = field(repr=False)
    _provider: Provider[AsyncOpenAI] = field(repr=False)

    @overload
    def __init__(
        self,
        model_name: OpenAIModelName,
        *,
        provider: Literal[
            'azure',
            'deepseek',
            'cerebras',
            'fireworks',
            'github',
            'grok',
            'heroku',
            'moonshotai',
            'ollama',
            'openai',
            'openai-chat',
            'openrouter',
            'together',
            'vercel',
            'litellm',
        ]
        | Provider[AsyncOpenAI] = 'openai',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ) -> None: ...

    @deprecated('Set the `system_prompt_role` in the `OpenAIModelProfile` instead.')
    @overload
    def __init__(
        self,
        model_name: OpenAIModelName,
        *,
        provider: Literal[
            'azure',
            'deepseek',
            'cerebras',
            'fireworks',
            'github',
            'grok',
            'heroku',
            'moonshotai',
            'ollama',
            'openai',
            'openai-chat',
            'openrouter',
            'together',
            'vercel',
            'litellm',
        ]
        | Provider[AsyncOpenAI] = 'openai',
        profile: ModelProfileSpec | None = None,
        system_prompt_role: OpenAISystemPromptRole | None = None,
        settings: ModelSettings | None = None,
    ) -> None: ...

    def __init__(
        self,
        model_name: OpenAIModelName,
        *,
        provider: Literal[
            'azure',
            'deepseek',
            'cerebras',
            'fireworks',
            'github',
            'grok',
            'heroku',
            'moonshotai',
            'ollama',
            'openai',
            'openai-chat',
            'openrouter',
            'together',
            'vercel',
            'litellm',
        ]
        | Provider[AsyncOpenAI] = 'openai',
        profile: ModelProfileSpec | None = None,
        system_prompt_role: OpenAISystemPromptRole | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize an OpenAI model.

        Args:
            model_name: The name of the OpenAI model to use. List of model names available
                [here](https://github.com/openai/openai-python/blob/v1.54.3/src/openai/types/chat_model.py#L7)
                (Unfortunately, despite being ask to do so, OpenAI do not provide `.inv` files for their API).
            provider: The provider to use. Defaults to `'openai'`.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            system_prompt_role: The role to use for the system prompt message. If not provided, defaults to `'system'`.
                In the future, this may be inferred from the model name.
            settings: Default model settings for this model instance.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider(provider)
        self._provider = provider
        self.client = provider.client

        super().__init__(settings=settings, profile=profile or provider.model_profile)

        if system_prompt_role is not None:
            self.profile = OpenAIModelProfile(openai_system_prompt_role=system_prompt_role).update(self.profile)

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def model_name(self) -> OpenAIModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The model provider."""
        return self._provider.name

    @property
    @deprecated('Set the `system_prompt_role` in the `OpenAIModelProfile` instead.')
    def system_prompt_role(self) -> OpenAISystemPromptRole | None:
        return OpenAIModelProfile.from_profile(self.profile).openai_system_prompt_role

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        response = await self._completions_create(
            messages, False, cast(OpenAIChatModelSettings, model_settings or {}), model_request_parameters
        )
        model_response = self._process_response(response)
        return model_response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        check_allow_model_requests()
        response = await self._completions_create(
            messages, True, cast(OpenAIChatModelSettings, model_settings or {}), model_request_parameters
        )
        async with response:
            yield await self._process_streamed_response(response, model_request_parameters)

    @overload
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[True],
        model_settings: OpenAIChatModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncStream[ChatCompletionChunk]: ...

    @overload
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: OpenAIChatModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> chat.ChatCompletion: ...

    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: OpenAIChatModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> chat.ChatCompletion | AsyncStream[ChatCompletionChunk]:
        return await _compat_completions_create(
            self,
            messages,
            stream,
            model_settings,
            model_request_parameters,
        )

    _process_response = partialmethod(
        process_response,
        map_usage_fn=map_usage,
        finish_reason_map=_CHAT_FINISH_REASON_MAP,
    )

    _process_streamed_response = partialmethod(
        process_streamed_response,
        map_usage_fn=map_usage,
        finish_reason_map=_CHAT_FINISH_REASON_MAP,
    )

    def _map_tool_definition(self, f: ToolDefinition) -> chat.ChatCompletionToolParam:
        return map_tool_definition(self.profile, f)

    async def _map_user_message(self, message: ModelRequest) -> AsyncIterable[chat.ChatCompletionMessageParam]:
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                system_prompt_role = OpenAIModelProfile.from_profile(self.profile).openai_system_prompt_role
                if system_prompt_role == 'developer':
                    yield chat.ChatCompletionDeveloperMessageParam(role='developer', content=part.content)
                elif system_prompt_role == 'user':
                    yield chat.ChatCompletionUserMessageParam(role='user', content=part.content)
                else:
                    yield chat.ChatCompletionSystemMessageParam(role='system', content=part.content)
            elif isinstance(part, UserPromptPart):
                yield await self._map_user_prompt(part)
            elif isinstance(part, ToolReturnPart):
                yield chat.ChatCompletionToolMessageParam(
                    role='tool',
                    tool_call_id=_guard_tool_call_id(t=part),
                    content=part.model_response_str(),
                )
            elif isinstance(part, RetryPromptPart):
                if part.tool_name is None:
                    yield chat.ChatCompletionUserMessageParam(role='user', content=part.model_response())
                else:
                    yield chat.ChatCompletionToolMessageParam(
                        role='tool',
                        tool_call_id=_guard_tool_call_id(t=part),
                        content=part.model_response(),
                    )
            else:
                assert_never(part)

    async def _map_user_prompt(self, part: UserPromptPart) -> chat.ChatCompletionUserMessageParam:  # noqa: C901
        content: str | list[ChatCompletionContentPartParam]
        if isinstance(part.content, str):
            content = part.content
        else:
            content = []
            for item in part.content:
                if isinstance(item, str):
                    content.append(ChatCompletionContentPartTextParam(text=item, type='text'))
                elif isinstance(item, ImageUrl):
                    image_url: ImageURL = {'url': item.url}
                    if metadata := item.vendor_metadata:
                        image_url['detail'] = metadata.get('detail', 'auto')
                    content.append(ChatCompletionContentPartImageParam(image_url=image_url, type='image_url'))
                elif isinstance(item, BinaryContent):
                    if self._is_text_like_media_type(item.media_type):
                        # Inline text-like binary content as a text block
                        content.append(
                            self._inline_text_file_part(
                                item.data.decode('utf-8'),
                                media_type=item.media_type,
                                identifier=item.identifier,
                            )
                        )
                    else:
                        base64_encoded = base64.b64encode(item.data).decode('utf-8')
                        if item.is_image:
                            image_url: ImageURL = {'url': f'data:{item.media_type};base64,{base64_encoded}'}
                            if metadata := item.vendor_metadata:
                                image_url['detail'] = metadata.get('detail', 'auto')
                            content.append(ChatCompletionContentPartImageParam(image_url=image_url, type='image_url'))
                        elif item.is_audio:
                            assert item.format in ('wav', 'mp3')
                            audio = InputAudio(data=base64_encoded, format=item.format)
                            content.append(
                                ChatCompletionContentPartInputAudioParam(input_audio=audio, type='input_audio')
                            )
                        elif item.is_document:
                            content.append(
                                File(
                                    file=FileFile(
                                        file_data=f'data:{item.media_type};base64,{base64_encoded}',
                                        filename=f'filename.{item.format}',
                                    ),
                                    type='file',
                                )
                            )
                        else:  # pragma: no cover
                            raise RuntimeError(f'Unsupported binary content type: {item.media_type}')
                elif isinstance(item, AudioUrl):
                    downloaded_item = await download_item(item, data_format='base64', type_format='extension')
                    assert downloaded_item['data_type'] in (
                        'wav',
                        'mp3',
                    ), f'Unsupported audio format: {downloaded_item["data_type"]}'
                    audio = InputAudio(data=downloaded_item['data'], format=downloaded_item['data_type'])
                    content.append(ChatCompletionContentPartInputAudioParam(input_audio=audio, type='input_audio'))
                elif isinstance(item, DocumentUrl):
                    if self._is_text_like_media_type(item.media_type):
                        downloaded_text = await download_item(item, data_format='text')
                        content.append(
                            self._inline_text_file_part(
                                downloaded_text['data'],
                                media_type=item.media_type,
                                identifier=item.identifier,
                            )
                        )
                    else:
                        downloaded_item = await download_item(item, data_format='base64_uri', type_format='extension')
                        content.append(
                            File(
                                file=FileFile(
                                    file_data=downloaded_item['data'],
                                    filename=f'filename.{downloaded_item["data_type"]}',
                                ),
                                type='file',
                            )
                        )
                elif isinstance(item, VideoUrl):  # pragma: no cover
                    raise NotImplementedError('VideoUrl is not supported for OpenAI')
                else:
                    assert_never(item)
        return chat.ChatCompletionUserMessageParam(role='user', content=content)

    @staticmethod
    def _is_text_like_media_type(media_type: str) -> bool:
        return (
            media_type.startswith('text/')
            or media_type == 'application/json'
            or media_type.endswith('+json')
            or media_type == 'application/xml'
            or media_type.endswith('+xml')
            or media_type in ('application/x-yaml', 'application/yaml')
        )

    @staticmethod
    def _inline_text_file_part(text: str, *, media_type: str, identifier: str) -> ChatCompletionContentPartTextParam:
        text = '\n'.join(
            [
                f'-----BEGIN FILE id="{identifier}" type="{media_type}"-----',
                text,
                f'-----END FILE id="{identifier}"-----',
            ]
        )
        return ChatCompletionContentPartTextParam(text=text, type='text')


@deprecated(
    '`OpenAIModel` was renamed to `OpenAIChatModel` to clearly distinguish it from `OpenAIResponsesModel` which '
    "uses OpenAI's newer Responses API. Use that unless you're using an OpenAI Chat Completions-compatible API, or "
    "require a feature that the Responses API doesn't support yet like audio."
)
@dataclass(init=False)
class OpenAIModel(OpenAIChatModel):
    """Deprecated alias for `OpenAIChatModel`."""


@dataclass(init=False)
class OpenAIResponsesModel(Model):
    """A model that uses the OpenAI Responses API.

    The [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) is the
    new API for OpenAI models.

    If you are interested in the differences between the Responses API and the Chat Completions API,
    see the [OpenAI API docs](https://platform.openai.com/docs/guides/responses-vs-chat-completions).
    """

    client: AsyncOpenAI = field(repr=False)

    _model_name: OpenAIModelName = field(repr=False)
    _provider: Provider[AsyncOpenAI] = field(repr=False)

    def __init__(
        self,
        model_name: OpenAIModelName,
        *,
        provider: Literal['openai', 'deepseek', 'azure', 'openrouter', 'grok', 'fireworks', 'together']
        | Provider[AsyncOpenAI] = 'openai',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize an OpenAI Responses model.

        Args:
            model_name: The name of the OpenAI model to use.
            provider: The provider to use. Defaults to `'openai'`.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: Default model settings for this model instance.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider(provider)
        self._provider = provider
        self.client = provider.client

        super().__init__(settings=settings, profile=profile or provider.model_profile)

    @property
    def model_name(self) -> OpenAIModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The model provider."""
        return self._provider.name

    async def request(
        self,
        messages: list[ModelRequest | ModelResponse],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        response = await responses_create(
            self,
            messages,
            False,
            cast(OpenAIResponsesModelSettings, model_settings or {}),
            model_request_parameters,
        )
        return process_responses_response(
            self,
            response,
            map_usage_fn=map_usage,
            finish_reason_map=cast(dict[str | ResponseStatus, FinishReason], _RESPONSES_FINISH_REASON_MAP),
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        check_allow_model_requests()
        response = await responses_create(
            self,
            messages,
            True,
            cast(OpenAIResponsesModelSettings, model_settings or {}),
            model_request_parameters,
        )
        async with response:
            yield await self._process_streamed_response(response, model_request_parameters)

    async def _process_streamed_response(
        self,
        response: AsyncStream[responses.ResponseStreamEvent],
        model_request_parameters: ModelRequestParameters,
    ) -> OpenAIResponsesStreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):  # pragma: no cover
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')

        assert isinstance(first_chunk, responses.ResponseCreatedEvent)
        return OpenAIResponsesStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=first_chunk.response.model,
            _response=peekable_response,
            _timestamp=number_to_datetime(first_chunk.response.created_at),
            _provider_name=self._provider.name,
        )

    def _get_reasoning(self, model_settings: OpenAIResponsesModelSettings) -> Reasoning | NotGiven:  # pragma: no cover
        return get_responses_reasoning(model_settings)

    def _get_tools(
        self, model_request_parameters: ModelRequestParameters
    ) -> list[responses.FunctionToolParam]:  # pragma: no cover
        return get_responses_tools(self.profile, list(model_request_parameters.tool_defs.values()))

    def _get_builtin_tools(
        self, model_request_parameters: ModelRequestParameters
    ) -> list[responses.ToolParam]:  # pragma: no cover
        return get_responses_builtin_tools(self.profile, model_request_parameters.builtin_tools)

    def _map_tool_definition(self, f: ToolDefinition) -> responses.FunctionToolParam:
        return map_responses_tool_definition(self.profile, f)

    def _get_previous_response_id_and_new_messages(
        self, messages: list[ModelMessage]
    ) -> tuple[str | None, list[ModelMessage]]:
        return get_responses_previous_response_id_and_new_messages(messages, self.system)

    async def _map_messages(
        self, messages: list[ModelMessage], model_settings: OpenAIResponsesModelSettings
    ) -> tuple[str | NotGiven, list[responses.ResponseInputItemParam]]:
        return await map_responses_messages(
            self.profile,
            self.system,
            messages,
            model_settings,
            map_user_prompt_fn=map_responses_user_prompt,
            get_instructions_fn=self._get_instructions,
        )

    def _map_json_schema(
        self, o: OutputObjectDefinition
    ) -> responses.ResponseFormatTextJSONSchemaConfigParam:  # pragma: no cover
        return map_responses_json_schema(self.profile, o)


@dataclass
class OpenAIResponsesStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for OpenAI Responses API."""

    _model_name: OpenAIModelName
    _response: AsyncIterable[responses.ResponseStreamEvent]
    _timestamp: datetime
    _provider_name: str

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:  # noqa: C901
        async for chunk in self._response:
            # NOTE: You can inspect the builtin tools used checking the `ResponseCompletedEvent`.
            if isinstance(chunk, responses.ResponseCompletedEvent):
                self._usage += map_usage(chunk.response)

                raw_finish_reason = (
                    details.reason if (details := chunk.response.incomplete_details) else chunk.response.status
                )
                if raw_finish_reason:  # pragma: no branch
                    self.provider_details = {'finish_reason': raw_finish_reason}
                    self.finish_reason = _RESPONSES_FINISH_REASON_MAP.get(raw_finish_reason)

            elif isinstance(chunk, responses.ResponseContentPartAddedEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseContentPartDoneEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseCreatedEvent):
                if chunk.response.id:  # pragma: no branch
                    self.provider_response_id = chunk.response.id

            elif isinstance(chunk, responses.ResponseFailedEvent):  # pragma: no cover
                self._usage += map_usage(chunk.response)

            elif isinstance(chunk, responses.ResponseFunctionCallArgumentsDeltaEvent):
                maybe_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=chunk.item_id,
                    args=chunk.delta,
                )
                if maybe_event is not None:  # pragma: no branch
                    yield maybe_event

            elif isinstance(chunk, responses.ResponseFunctionCallArgumentsDoneEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseIncompleteEvent):  # pragma: no cover
                self._usage += map_usage(chunk.response)

            elif isinstance(chunk, responses.ResponseInProgressEvent):
                self._usage += map_usage(chunk.response)

            elif isinstance(chunk, responses.ResponseOutputItemAddedEvent):
                if isinstance(chunk.item, responses.ResponseFunctionToolCall):
                    yield self._parts_manager.handle_tool_call_part(
                        vendor_part_id=chunk.item.id,
                        tool_name=chunk.item.name,
                        args=chunk.item.arguments,
                        tool_call_id=combine_tool_call_ids(chunk.item.call_id, chunk.item.id),
                    )
                elif isinstance(chunk.item, responses.ResponseReasoningItem):
                    pass
                elif isinstance(chunk.item, responses.ResponseOutputMessage):
                    pass
                elif isinstance(chunk.item, responses.ResponseFunctionWebSearch):
                    call_part, _ = map_web_search_tool_call(chunk.item, self.provider_name)
                    yield self._parts_manager.handle_builtin_tool_call_part(
                        vendor_part_id=f'{chunk.item.id}-call', part=replace(call_part, args=None)
                    )
                elif isinstance(chunk.item, responses.ResponseCodeInterpreterToolCall):
                    call_part, _ = map_code_interpreter_tool_call(chunk.item, self.provider_name)

                    args_json = call_part.args_as_json_str()
                    # Drop the final `"}` so that we can add code deltas
                    args_json_delta = args_json[:-2]
                    assert args_json_delta.endswith('code":"')

                    yield self._parts_manager.handle_builtin_tool_call_part(
                        vendor_part_id=f'{chunk.item.id}-call', part=replace(call_part, args=None)
                    )
                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=f'{chunk.item.id}-call',
                        args=args_json_delta,
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event
                else:
                    warnings.warn(  # pragma: no cover
                        f'Handling of this item type is not yet implemented. Please report on our GitHub: {chunk}',
                        UserWarning,
                    )

            elif isinstance(chunk, responses.ResponseOutputItemDoneEvent):
                if isinstance(chunk.item, responses.ResponseReasoningItem):
                    if signature := chunk.item.encrypted_content:  # pragma: no branch
                        # Add the signature to the part corresponding to the first summary item
                        yield self._parts_manager.handle_thinking_delta(
                            vendor_part_id=f'{chunk.item.id}-0',
                            id=chunk.item.id,
                            signature=signature,
                            provider_name=self.provider_name,
                        )
                elif isinstance(chunk.item, responses.ResponseCodeInterpreterToolCall):
                    _, return_part = map_code_interpreter_tool_call(chunk.item, self.provider_name)
                    yield self._parts_manager.handle_builtin_tool_return_part(
                        vendor_part_id=f'{chunk.item.id}-return', part=return_part
                    )
                elif isinstance(chunk.item, responses.ResponseFunctionWebSearch):
                    call_part, return_part = map_web_search_tool_call(chunk.item, self.provider_name)

                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=f'{chunk.item.id}-call',
                        args=call_part.args,
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event

                    yield self._parts_manager.handle_builtin_tool_return_part(
                        vendor_part_id=f'{chunk.item.id}-return', part=return_part
                    )

            elif isinstance(chunk, responses.ResponseReasoningSummaryPartAddedEvent):
                yield self._parts_manager.handle_thinking_delta(
                    vendor_part_id=f'{chunk.item_id}-{chunk.summary_index}',
                    content=chunk.part.text,
                    id=chunk.item_id,
                )

            elif isinstance(chunk, responses.ResponseReasoningSummaryPartDoneEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseReasoningSummaryTextDoneEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseReasoningSummaryTextDeltaEvent):
                yield self._parts_manager.handle_thinking_delta(
                    vendor_part_id=f'{chunk.item_id}-{chunk.summary_index}',
                    content=chunk.delta,
                    id=chunk.item_id,
                )

            # TODO(Marcelo): We should support annotations in the future.
            elif isinstance(chunk, responses.ResponseOutputTextAnnotationAddedEvent):  # pragma: no cover
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseTextDeltaEvent):
                maybe_event = self._parts_manager.handle_text_delta(
                    vendor_part_id=chunk.item_id, content=chunk.delta, id=chunk.item_id
                )
                if maybe_event is not None:  # pragma: no branch
                    yield maybe_event

            elif isinstance(chunk, responses.ResponseTextDoneEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseWebSearchCallInProgressEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseWebSearchCallSearchingEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseWebSearchCallCompletedEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseAudioDeltaEvent):  # pragma: lax no cover
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseCodeInterpreterCallCodeDeltaEvent):
                json_args_delta = to_json(chunk.delta).decode()[1:-1]  # Drop the surrounding `"`
                maybe_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=f'{chunk.item_id}-call',
                    args=json_args_delta,
                )
                if maybe_event is not None:  # pragma: no branch
                    yield maybe_event

            elif isinstance(chunk, responses.ResponseCodeInterpreterCallCodeDoneEvent):
                maybe_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=f'{chunk.item_id}-call',
                    args='"}',
                )
                if maybe_event is not None:  # pragma: no branch
                    yield maybe_event

            elif isinstance(chunk, responses.ResponseCodeInterpreterCallCompletedEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseCodeInterpreterCallInProgressEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseCodeInterpreterCallInterpretingEvent):
                pass  # there's nothing we need to do here

            else:  # pragma: no cover
                warnings.warn(
                    f'Handling of this event type is not yet implemented. Please report on our GitHub: {chunk}',
                    UserWarning,
                )

    @property
    def model_name(self) -> OpenAIModelName:
        """Get the model name of the response."""
        return self._model_name

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self._provider_name

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._timestamp
