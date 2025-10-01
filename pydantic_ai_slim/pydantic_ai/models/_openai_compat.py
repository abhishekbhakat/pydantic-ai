"""Shared OpenAI compatibility helpers (in-progress).

This module is a working scaffold. Implementations will be ported in small,
covered steps from `_openai_compat_ref.py` to preserve coverage.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Literal, cast, overload

from pydantic import ValidationError
from typing_extensions import assert_never

from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    DocumentUrl,
    FinishReason,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    PartStartEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from .. import UnexpectedModelBehavior, _utils, usage
from .._output import DEFAULT_OUTPUT_TOOL_NAME, OutputObjectDefinition
from .._thinking_part import split_content_into_text_and_thinking
from .._utils import guard_tool_call_id as _guard_tool_call_id, now_utc as _now_utc, number_to_datetime
from ..builtin_tools import CodeExecutionTool, WebSearchTool
from ..exceptions import UserError
from ..profiles import ModelProfile
from ..profiles.openai import OpenAIModelProfile
from ..settings import ModelSettings
from ..tools import ToolDefinition
from . import ModelRequestParameters, StreamedResponse, download_item, get_user_agent

try:
    from openai import NOT_GIVEN, APIStatusError, AsyncStream, NotGiven
    from openai.types import chat, responses
    from openai.types.chat import (
        ChatCompletionChunk,
        ChatCompletionMessageCustomToolCall,
        ChatCompletionMessageFunctionToolCall,
    )
    from openai.types.chat.completion_create_params import ResponseFormat, WebSearchOptions
    from openai.types.responses.response_input_param import FunctionCallOutput, Message
    from openai.types.responses.response_reasoning_item_param import Summary
    from openai.types.responses.response_status import ResponseStatus
    from openai.types.shared_params import Reasoning
except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` to use the OpenAI model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

__all__ = (
    'OpenAICompatStreamedResponse',
    'combine_tool_call_ids',
    'completions_create',
    'get_responses_builtin_tools',
    'get_responses_previous_response_id_and_new_messages',
    'get_responses_reasoning',
    'get_responses_tools',
    'map_code_interpreter_tool_call',
    'map_messages',
    'map_responses_json_schema',
    'map_responses_messages',
    'map_responses_tool_definition',
    'map_responses_user_prompt',
    'map_tool_definition',
    'map_usage',
    'map_web_search_tool_call',
    'process_response',
    'process_responses_response',
    'process_streamed_response',
    'responses_create',
    'split_combined_tool_call_id',
)


def _map_tool_call(t: ToolCallPart) -> Any:
    """Map a ToolCallPart to OpenAI ChatCompletionMessageFunctionToolCallParam."""
    return {
        'id': _guard_tool_call_id(t=t),
        'type': 'function',
        'function': {'name': t.tool_name, 'arguments': t.args_as_json_str()},
    }


def map_tool_definition(model_profile: ModelProfile, f: ToolDefinition) -> Any:
    """Map a ToolDefinition to OpenAI ChatCompletionToolParam."""
    tool_param: dict[str, Any] = {
        'type': 'function',
        'function': {
            'name': f.name,
            'description': f.description or '',
            'parameters': f.parameters_json_schema,
        },
    }
    if f.strict and OpenAIModelProfile.from_profile(model_profile).openai_supports_strict_tool_definition:
        tool_param['function']['strict'] = f.strict
    return tool_param


async def map_messages(model: Any, messages: list[ModelMessage]) -> list[Any]:
    """Async mapping of internal ModelMessage list to OpenAI chat messages."""
    openai_messages: list[Any] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            async for item in model._map_user_message(message):
                openai_messages.append(item)
        elif isinstance(message, ModelResponse):
            texts: list[str] = []
            tool_calls: list[Any] = []
            for item in message.parts:
                if isinstance(item, TextPart):
                    texts.append(item.content)
                elif isinstance(item, ToolCallPart):
                    tool_calls.append(_map_tool_call(item))
            message_param: dict[str, Any] = {'role': 'assistant'}
            if texts:
                message_param['content'] = '\n\n'.join(texts)
            else:
                message_param['content'] = None
            if tool_calls:
                message_param['tool_calls'] = tool_calls
            openai_messages.append(message_param)
        else:
            assert_never(message)

    return openai_messages


def get_tools(model_profile: ModelProfile, tool_defs: dict[str, ToolDefinition]) -> list[Any]:
    """Get OpenAI tools from tool definitions."""
    return [map_tool_definition(model_profile, r) for r in tool_defs.values()]


def _map_json_schema(model_profile: ModelProfile, o: OutputObjectDefinition) -> ResponseFormat:
    """Map an OutputObjectDefinition to OpenAI ResponseFormatJSONSchema."""
    response_format_param: ResponseFormat = {
        'type': 'json_schema',
        'json_schema': {'name': o.name or 'output', 'schema': o.json_schema},
    }
    if o.description:
        response_format_param['json_schema']['description'] = o.description
    profile = OpenAIModelProfile.from_profile(model_profile)
    if profile.openai_supports_strict_tool_definition:  # pragma: no branch
        response_format_param['json_schema']['strict'] = bool(o.strict)
    return response_format_param


def _get_web_search_options(model_profile: ModelProfile, builtin_tools: list[Any]) -> WebSearchOptions | None:
    """Extract WebSearchOptions from builtin_tools if WebSearchTool is present."""
    for tool in builtin_tools:
        if tool.__class__.__name__ == 'WebSearchTool':
            if not OpenAIModelProfile.from_profile(model_profile).openai_chat_supports_web_search:
                raise UserError(
                    f'WebSearchTool is not supported with `OpenAIChatModel` and model {getattr(model_profile, "model_name", None) or "<unknown>"!r}. '
                    f'Please use `OpenAIResponsesModel` instead.'
                )
            if tool.user_location:
                from openai.types.chat.completion_create_params import (
                    WebSearchOptionsUserLocation,
                    WebSearchOptionsUserLocationApproximate,
                )

                return WebSearchOptions(
                    search_context_size=tool.search_context_size,
                    user_location=WebSearchOptionsUserLocation(
                        type='approximate',
                        approximate=WebSearchOptionsUserLocationApproximate(**tool.user_location),
                    ),
                )
            return WebSearchOptions(search_context_size=tool.search_context_size)
        else:
            raise UserError(
                f'`{tool.__class__.__name__}` is not supported by `OpenAIChatModel`. If it should be, please file an issue.'
            )
    return None


@overload
async def completions_create(
    model: Any,
    messages: list[ModelMessage],
    stream: Literal[True],
    model_settings: ModelSettings,
    model_request_parameters: ModelRequestParameters,
) -> AsyncStream[ChatCompletionChunk]: ...


@overload
async def completions_create(
    model: Any,
    messages: list[ModelMessage],
    stream: Literal[False],
    model_settings: ModelSettings,
    model_request_parameters: ModelRequestParameters,
) -> chat.ChatCompletion: ...


async def completions_create(
    model: Any,
    messages: list[ModelMessage],
    stream: bool,
    model_settings: ModelSettings,
    model_request_parameters: ModelRequestParameters,
) -> chat.ChatCompletion | AsyncStream[ChatCompletionChunk]:
    """Create a chat completion using OpenAI SDK with compat helpers.

    Handles tool mapping, response-format mapping, unsupported-setting pruning,
    and SDK invocation with error translation.
    """
    tools = get_tools(model.profile, model_request_parameters.tool_defs)
    web_search_options = _get_web_search_options(model.profile, model_request_parameters.builtin_tools)

    if not tools:
        tool_choice: Literal['none', 'required', 'auto'] | None = None
    elif (
        not model_request_parameters.allow_text_output
        and OpenAIModelProfile.from_profile(model.profile).openai_supports_tool_choice_required
    ):
        tool_choice = 'required'
    else:
        tool_choice = 'auto'

    openai_messages = await map_messages(model, messages)

    response_format: ResponseFormat | None = None
    if model_request_parameters.output_mode == 'native':
        output_object = model_request_parameters.output_object
        assert output_object is not None
        response_format = _map_json_schema(model.profile, output_object)
    elif (
        model_request_parameters.output_mode == 'prompted' and model.profile.supports_json_object_output
    ):  # pragma: no branch
        response_format = {'type': 'json_object'}

    unsupported_model_settings = OpenAIModelProfile.from_profile(model.profile).openai_unsupported_model_settings
    for setting in unsupported_model_settings:
        model_settings.pop(setting, None)

    try:
        extra_headers = model_settings.get('extra_headers', {})
        extra_headers.setdefault('User-Agent', get_user_agent())
        return await model.client.chat.completions.create(
            model=model._model_name,
            messages=openai_messages,
            parallel_tool_calls=model_settings.get('parallel_tool_calls', NOT_GIVEN),
            tools=tools or NOT_GIVEN,
            tool_choice=tool_choice or NOT_GIVEN,
            stream=stream,
            stream_options={'include_usage': True} if stream else NOT_GIVEN,
            stop=model_settings.get('stop_sequences', NOT_GIVEN),
            max_completion_tokens=model_settings.get('max_tokens', NOT_GIVEN),
            timeout=model_settings.get('timeout', NOT_GIVEN),
            response_format=response_format or NOT_GIVEN,
            seed=model_settings.get('seed', NOT_GIVEN),
            reasoning_effort=model_settings.get('openai_reasoning_effort', NOT_GIVEN),
            user=model_settings.get('openai_user', NOT_GIVEN),
            web_search_options=web_search_options or NOT_GIVEN,
            service_tier=model_settings.get('openai_service_tier', NOT_GIVEN),
            prediction=model_settings.get('openai_prediction', NOT_GIVEN),
            temperature=model_settings.get('temperature', NOT_GIVEN),
            top_p=model_settings.get('top_p', NOT_GIVEN),
            presence_penalty=model_settings.get('presence_penalty', NOT_GIVEN),
            frequency_penalty=model_settings.get('frequency_penalty', NOT_GIVEN),
            logit_bias=model_settings.get('logit_bias', NOT_GIVEN),
            logprobs=model_settings.get('openai_logprobs', NOT_GIVEN),
            top_logprobs=model_settings.get('openai_top_logprobs', NOT_GIVEN),
            extra_headers=extra_headers,
            extra_body=model_settings.get('extra_body'),
        )
    except APIStatusError as e:
        if (status_code := e.status_code) >= 400:
            from .. import ModelHTTPError

            raise ModelHTTPError(status_code=status_code, model_name=model.model_name, body=e.body) from e
        raise  # pragma: lax no cover


def process_response(
    model: Any,
    response: chat.ChatCompletion | str,
    *,
    map_usage_fn: Callable[[chat.ChatCompletion], usage.RequestUsage],
    finish_reason_map: Mapping[str, FinishReason],
) -> ModelResponse:
    """Process a non-streamed chat completion response into a ModelResponse."""
    if not isinstance(response, chat.ChatCompletion):
        raise UnexpectedModelBehavior('Invalid response from OpenAI chat completions endpoint, expected JSON data')

    if response.created:
        timestamp = number_to_datetime(response.created)
    else:
        timestamp = _now_utc()
        response.created = int(timestamp.timestamp())

    # Workaround for local Ollama which sometimes returns a `None` finish reason.
    if response.choices and (choice := response.choices[0]) and choice.finish_reason is None:  # pyright: ignore[reportUnnecessaryComparison]
        choice.finish_reason = 'stop'

    try:
        response = chat.ChatCompletion.model_validate(response.model_dump())
    except ValidationError as e:  # pragma: no cover
        raise UnexpectedModelBehavior(f'Invalid response from OpenAI chat completions endpoint: {e}') from e

    choice = response.choices[0]
    items: list[ModelResponsePart] = []

    # OpenRouter uses 'reasoning', OpenAI previously used 'reasoning_content' (removed Feb 2025)
    reasoning_content = getattr(choice.message, 'reasoning', None) or getattr(choice.message, 'reasoning_content', None)
    if reasoning_content:
        items.append(ThinkingPart(id='reasoning_content', content=reasoning_content, provider_name=model.system))

    vendor_details: dict[str, Any] = {}

    if choice.logprobs is not None and choice.logprobs.content:
        vendor_details['logprobs'] = [
            {
                'token': lp.token,
                'bytes': lp.bytes,
                'logprob': lp.logprob,
                'top_logprobs': [
                    {'token': tlp.token, 'bytes': tlp.bytes, 'logprob': tlp.logprob} for tlp in lp.top_logprobs
                ],
            }
            for lp in choice.logprobs.content
        ]

    if choice.message.content is not None:
        items.extend(
            (replace(part, id='content', provider_name=model.system) if isinstance(part, ThinkingPart) else part)
            for part in split_content_into_text_and_thinking(choice.message.content, model.profile.thinking_tags)
        )

    if choice.message.tool_calls:
        for tool_call in choice.message.tool_calls:
            if isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
                part = ToolCallPart(tool_call.function.name, tool_call.function.arguments, tool_call_id=tool_call.id)
            elif isinstance(tool_call, ChatCompletionMessageCustomToolCall):  # pragma: no cover
                raise RuntimeError('Custom tool calls are not supported')
            else:
                assert_never(tool_call)
            part.tool_call_id = _guard_tool_call_id(part)
            items.append(part)

    raw_finish_reason = choice.finish_reason
    vendor_details['finish_reason'] = raw_finish_reason
    finish_reason = finish_reason_map.get(raw_finish_reason)

    return ModelResponse(
        parts=items,
        usage=map_usage_fn(response),
        model_name=response.model,
        timestamp=timestamp,
        provider_details=vendor_details or None,
        provider_response_id=response.id,
        provider_name=model.system,
        finish_reason=finish_reason,
    )


async def process_streamed_response(
    model: Any,
    response: AsyncStream[ChatCompletionChunk],
    model_request_parameters: ModelRequestParameters,
    *,
    map_usage_fn: Callable[[ChatCompletionChunk], usage.RequestUsage],
    finish_reason_map: Mapping[str, FinishReason],
) -> OpenAICompatStreamedResponse:
    """Wrap a streamed chat completion response with compat handling."""
    peekable_response = _utils.PeekableAsyncStream(response)
    first_chunk = await peekable_response.peek()
    if isinstance(first_chunk, _utils.Unset):  # pragma: no cover
        raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')

    model_name = first_chunk.model or model.model_name

    return OpenAICompatStreamedResponse(
        model_request_parameters=model_request_parameters,
        _model_name=model_name,
        _model_profile=model.profile,
        _response=peekable_response,
        _timestamp=number_to_datetime(first_chunk.created),
        _provider_name=model.system,
        _map_usage_fn=map_usage_fn,
        _finish_reason_map=finish_reason_map,
    )


@dataclass
class OpenAICompatStreamedResponse(StreamedResponse):
    """Streaming response wrapper for OpenAI chat completions."""

    model_request_parameters: ModelRequestParameters
    _model_name: str
    _model_profile: ModelProfile
    _response: AsyncIterable[ChatCompletionChunk]
    _timestamp: datetime
    _provider_name: str
    _map_usage_fn: Callable[[ChatCompletionChunk], usage.RequestUsage] = field(repr=False)
    _finish_reason_map: Mapping[str, FinishReason] = field(repr=False)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        async for chunk in self._response:
            self._usage += self._map_usage_fn(chunk)

            if chunk.id:  # pragma: no branch
                self.provider_response_id = chunk.id

            if chunk.model:
                self._model_name = chunk.model

            try:
                choice = chunk.choices[0]
            except IndexError:
                continue

            if choice.delta is None:  # pyright: ignore[reportUnnecessaryComparison]
                continue

            if raw_finish_reason := choice.finish_reason:
                self.provider_details = {'finish_reason': raw_finish_reason}
                self.finish_reason = self._finish_reason_map.get(raw_finish_reason)

            content = choice.delta.content
            if content is not None:
                maybe_event = self._parts_manager.handle_text_delta(
                    vendor_part_id='content',
                    content=content,
                    thinking_tags=self._model_profile.thinking_tags,
                    ignore_leading_whitespace=self._model_profile.ignore_streamed_leading_whitespace,
                )
                if maybe_event is not None:
                    if isinstance(maybe_event, PartStartEvent) and isinstance(maybe_event.part, ThinkingPart):
                        maybe_event.part.id = 'content'
                        maybe_event.part.provider_name = self.provider_name
                    yield maybe_event

            # OpenRouter uses 'reasoning', OpenAI previously used 'reasoning_content' (removed Feb 2025)
            reasoning_content = getattr(choice.delta, 'reasoning', None) or getattr(
                choice.delta, 'reasoning_content', None
            )
            if reasoning_content:
                yield self._parts_manager.handle_thinking_delta(
                    vendor_part_id='reasoning_content',
                    id='reasoning_content',
                    content=reasoning_content,
                    provider_name=self.provider_name,
                )

            for dtc in choice.delta.tool_calls or []:
                maybe_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=dtc.index,
                    tool_name=dtc.function and dtc.function.name,
                    args=dtc.function and dtc.function.arguments,
                    tool_call_id=dtc.id,
                )
                if maybe_event is not None:
                    yield maybe_event

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def timestamp(self) -> datetime:
        return self._timestamp


def combine_tool_call_ids(call_id: str, id: str | None) -> str:
    """Combine tool call IDs used by OpenAI Responses API."""
    return f'{call_id}|{id}' if id else call_id


def split_combined_tool_call_id(combined_id: str) -> tuple[str, str | None]:
    """Split combined tool call IDs used by OpenAI Responses API."""
    if '|' in combined_id:
        call_id, id = combined_id.split('|', 1)
        return call_id, id
    return combined_id, None  # pragma: no cover


def map_code_interpreter_tool_call(
    item: responses.ResponseCodeInterpreterToolCall, provider_name: str
) -> tuple[BuiltinToolCallPart, BuiltinToolReturnPart]:
    """Map code interpreter tool call into Pydantic AI builtin parts."""
    result: dict[str, Any] = {'status': item.status}
    if item.outputs:
        result['outputs'] = [output.model_dump(mode='json') for output in item.outputs]

    return (
        BuiltinToolCallPart(
            tool_name=CodeExecutionTool.kind,
            tool_call_id=item.id,
            args={'container_id': item.container_id, 'code': item.code},
            provider_name=provider_name,
        ),
        BuiltinToolReturnPart(
            tool_name=CodeExecutionTool.kind,
            tool_call_id=item.id,
            content=result,
            provider_name=provider_name,
        ),
    )


def map_web_search_tool_call(
    item: responses.ResponseFunctionWebSearch, provider_name: str
) -> tuple[BuiltinToolCallPart, BuiltinToolReturnPart]:
    """Map web search tool call into Pydantic AI builtin parts."""
    args: dict[str, Any] | None = None
    result: dict[str, Any] = {'status': item.status}

    if action := item.action:
        args = action.model_dump(mode='json')
        if sources := args.pop('sources', None):
            result['sources'] = sources

    return (
        BuiltinToolCallPart(
            tool_name=WebSearchTool.kind,
            tool_call_id=item.id,
            args=args,
            provider_name=provider_name,
        ),
        BuiltinToolReturnPart(
            tool_name=WebSearchTool.kind,
            tool_call_id=item.id,
            content=result,
            provider_name=provider_name,
        ),
    )


def map_usage(
    response: chat.ChatCompletion | ChatCompletionChunk | responses.Response,
) -> usage.RequestUsage:
    response_usage = response.usage
    if response_usage is None:
        return usage.RequestUsage()
    elif isinstance(response_usage, responses.ResponseUsage):
        details: dict[str, int] = {
            key: value
            for key, value in response_usage.model_dump(
                exclude_none=True,
                exclude={'input_tokens', 'output_tokens', 'total_tokens', 'completion_tokens_details'},
            ).items()
            if isinstance(value, int)
        }
        reasoning_tokens = 0
        for detail_name in ('completion_tokens_details', 'output_tokens_details'):
            detail = getattr(response_usage, detail_name, None)
            tokens = getattr(detail, 'reasoning_tokens', None)
            if tokens is not None:
                reasoning_tokens = tokens or 0
                break
        details['reasoning_tokens'] = reasoning_tokens
        cache_read_tokens = 0
        if getattr(response_usage, 'input_tokens_details', None) is not None:
            cache_read_tokens = response_usage.input_tokens_details.cached_tokens or 0

        return usage.RequestUsage(
            input_tokens=response_usage.input_tokens,
            output_tokens=response_usage.output_tokens,
            cache_read_tokens=cache_read_tokens,
            details=details,
        )
    else:
        details = {
            key: value
            for key, value in response_usage.model_dump(
                exclude_none=True,
                exclude={'prompt_tokens', 'completion_tokens', 'total_tokens'},
            ).items()
            if isinstance(value, int)
        }
        result = usage.RequestUsage(
            input_tokens=response_usage.prompt_tokens,
            output_tokens=response_usage.completion_tokens,
            details=details,
        )
        if response_usage.completion_tokens_details is not None:
            details.update(response_usage.completion_tokens_details.model_dump(exclude_none=True))
            result.output_audio_tokens = response_usage.completion_tokens_details.audio_tokens or 0
        if response_usage.prompt_tokens_details is not None:
            result.input_audio_tokens = response_usage.prompt_tokens_details.audio_tokens or 0
            result.cache_read_tokens = response_usage.prompt_tokens_details.cached_tokens or 0
        return result


def map_responses_json_schema(
    profile: ModelProfile, output: OutputObjectDefinition
) -> responses.ResponseFormatTextJSONSchemaConfigParam:
    param: responses.ResponseFormatTextJSONSchemaConfigParam = {
        'type': 'json_schema',
        'name': output.name or DEFAULT_OUTPUT_TOOL_NAME,
        'schema': output.json_schema,
    }
    if output.description:
        param['description'] = output.description
    if OpenAIModelProfile.from_profile(profile).openai_supports_strict_tool_definition:  # pragma: no branch
        param['strict'] = bool(output.strict)
    return param


def get_responses_reasoning(model_settings: Mapping[str, Any]) -> Reasoning | NotGiven:
    effort = model_settings.get('openai_reasoning_effort')
    summary = model_settings.get('openai_reasoning_summary')
    if effort is None and summary is None:
        return NOT_GIVEN
    return Reasoning(effort=effort, summary=summary)


def map_responses_tool_definition(profile: ModelProfile, tool: ToolDefinition) -> responses.FunctionToolParam:
    return {
        'name': tool.name,
        'parameters': tool.parameters_json_schema,
        'type': 'function',
        'description': tool.description,
        'strict': bool(tool.strict and OpenAIModelProfile.from_profile(profile).openai_supports_strict_tool_definition),
    }


def get_responses_tools(
    profile: ModelProfile, tool_defs: Sequence[ToolDefinition]
) -> list[responses.FunctionToolParam]:
    return [map_responses_tool_definition(profile, tool) for tool in tool_defs]


def get_responses_builtin_tools(profile: ModelProfile, builtin_tools: Sequence[Any]) -> list[responses.ToolParam]:
    tools: list[responses.ToolParam] = []
    for tool in builtin_tools:
        if isinstance(tool, WebSearchTool):
            web_search_tool = responses.WebSearchToolParam(
                type='web_search', search_context_size=tool.search_context_size
            )
            if tool.user_location:
                web_search_tool['user_location'] = responses.web_search_tool_param.UserLocation(
                    type='approximate', **tool.user_location
                )
            tools.append(web_search_tool)
        elif isinstance(tool, CodeExecutionTool):  # pragma: no branch
            tools.append({'type': 'code_interpreter', 'container': {'type': 'auto'}})
    return tools


def get_responses_previous_response_id_and_new_messages(
    messages: list[ModelMessage], system_name: str
) -> tuple[str | None, list[ModelMessage]]:
    previous_response_id = None
    trimmed: list[ModelMessage] = []
    for message in reversed(messages):
        if isinstance(message, ModelResponse) and message.provider_name == system_name:
            previous_response_id = message.provider_response_id
            break
        trimmed.append(message)
    if previous_response_id and trimmed:
        return previous_response_id, list(reversed(trimmed))
    return None, messages


async def map_responses_user_prompt(part: UserPromptPart) -> responses.EasyInputMessageParam:
    if isinstance(part.content, str):
        return responses.EasyInputMessageParam(role='user', content=part.content)
    items: list[responses.ResponseInputContentParam] = []
    for item in part.content:
        if isinstance(item, str):
            items.append(responses.ResponseInputTextParam(text=item, type='input_text'))
        elif isinstance(item, BinaryContent):
            base64_encoded = base64.b64encode(item.data).decode('utf-8')
            if item.is_image:
                detail = 'auto'
                if item.vendor_metadata and 'detail' in item.vendor_metadata:
                    detail = item.vendor_metadata['detail']
                items.append(
                    responses.ResponseInputImageParam(
                        image_url=f'data:{item.media_type};base64,{base64_encoded}',
                        type='input_image',
                        detail=detail,
                    )
                )
            elif item.is_document:
                items.append(
                    responses.ResponseInputFileParam(
                        type='input_file',
                        file_data=f'data:{item.media_type};base64,{base64_encoded}',
                        filename=f'filename.{item.format}',
                    )
                )
            elif item.is_audio:
                raise NotImplementedError('Audio as binary content is not supported for OpenAI Responses API.')
        elif isinstance(item, ImageUrl):
            detail = 'auto'
            if item.vendor_metadata and 'detail' in item.vendor_metadata:
                detail = item.vendor_metadata['detail']
            items.append(responses.ResponseInputImageParam(image_url=item.url, type='input_image', detail=detail))
        elif isinstance(item, AudioUrl):  # pragma: no cover
            downloaded = await download_item(item, data_format='base64_uri', type_format='extension')
            items.append(
                responses.ResponseInputFileParam(
                    type='input_file',
                    file_data=downloaded['data'],
                    filename=f'filename.{downloaded["data_type"]}',
                )
            )
        elif isinstance(item, DocumentUrl):
            downloaded = await download_item(item, data_format='base64_uri', type_format='extension')
            items.append(
                responses.ResponseInputFileParam(
                    type='input_file',
                    file_data=downloaded['data'],
                    filename=f'filename.{downloaded["data_type"]}',
                )
            )
    return responses.EasyInputMessageParam(role='user', content=items)


async def map_responses_messages(  # noqa: C901
    profile: ModelProfile,
    system_name: str,
    messages: list[ModelMessage],
    model_settings: Mapping[str, Any],
    *,
    map_user_prompt_fn: Callable[[UserPromptPart], Awaitable[responses.EasyInputMessageParam]],
    get_instructions_fn: Callable[[list[ModelMessage]], str | None],
) -> tuple[str | NotGiven, list[responses.ResponseInputItemParam]]:
    profile_data = OpenAIModelProfile.from_profile(profile)
    send_item_ids = model_settings.get(
        'openai_send_reasoning_ids', profile_data.openai_supports_encrypted_reasoning_content
    )
    openai_messages: list[responses.ResponseInputItemParam] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, SystemPromptPart):
                    openai_messages.append(responses.EasyInputMessageParam(role='system', content=part.content))
                elif isinstance(part, UserPromptPart):
                    openai_messages.append(await map_user_prompt_fn(part))
                elif isinstance(part, ToolReturnPart):
                    call_id = _guard_tool_call_id(part)
                    call_id, _ = split_combined_tool_call_id(call_id)
                    openai_messages.append(
                        FunctionCallOutput(
                            type='function_call_output',
                            call_id=call_id,
                            output=part.model_response_str(),
                        )
                    )
                elif isinstance(part, RetryPromptPart):
                    if part.tool_name is None:  # pragma: no cover
                        openai_messages.append(
                            Message(role='user', content=[{'type': 'input_text', 'text': part.model_response()}])
                        )
                    else:
                        call_id = _guard_tool_call_id(part)
                        call_id, _ = split_combined_tool_call_id(call_id)
                        openai_messages.append(
                            FunctionCallOutput(
                                type='function_call_output',
                                call_id=call_id,
                                output=part.model_response(),
                            )
                        )
                else:
                    assert_never(part)
        elif isinstance(message, ModelResponse):
            send_item_ids = send_item_ids and message.provider_name == system_name
            message_item: responses.ResponseOutputMessageParam | None = None
            reasoning_item: responses.ResponseReasoningItemParam | None = None
            web_search_item: responses.ResponseFunctionWebSearchParam | None = None
            code_interpreter_item: responses.ResponseCodeInterpreterToolCallParam | None = None
            for part in message.parts:
                if isinstance(part, TextPart):
                    if part.id and send_item_ids:
                        if message_item is None or message_item['id'] != part.id:  # pragma: no branch
                            message_item = responses.ResponseOutputMessageParam(
                                role='assistant', id=part.id, content=[], type='message', status='completed'
                            )
                            openai_messages.append(message_item)
                        message_item['content'] = [
                            *message_item['content'],
                            responses.ResponseOutputTextParam(text=part.content, type='output_text', annotations=[]),
                        ]
                    else:
                        openai_messages.append(responses.EasyInputMessageParam(role='assistant', content=part.content))
                elif isinstance(part, ToolCallPart):
                    call_id = _guard_tool_call_id(part)
                    call_id, part_id = split_combined_tool_call_id(call_id)
                    param = responses.ResponseFunctionToolCallParam(
                        name=part.tool_name,
                        arguments=part.args_as_json_str(),
                        call_id=call_id,
                        type='function_call',
                    )
                    if part_id and send_item_ids:
                        param['id'] = part_id
                    openai_messages.append(param)
                elif isinstance(part, BuiltinToolCallPart):
                    if part.provider_name == system_name:
                        if (
                            part.tool_name == CodeExecutionTool.kind
                            and part.tool_call_id
                            and (args := part.args_as_dict())
                            and (container_id := args.get('container_id'))
                        ):
                            code_interpreter_item = responses.ResponseCodeInterpreterToolCallParam(
                                id=part.tool_call_id,
                                code=args.get('code'),
                                container_id=container_id,
                                outputs=None,
                                status='completed',
                                type='code_interpreter_call',
                            )
                            openai_messages.append(code_interpreter_item)
                        elif (
                            part.tool_name == WebSearchTool.kind and part.tool_call_id and (args := part.args_as_dict())
                        ):  # pragma: no branch
                            web_search_item = responses.ResponseFunctionWebSearchParam(
                                id=part.tool_call_id,
                                action=cast(responses.response_function_web_search_param.Action, args),
                                status='completed',
                                type='web_search_call',
                            )
                            openai_messages.append(web_search_item)
                elif isinstance(part, BuiltinToolReturnPart):
                    if part.provider_name == system_name:
                        if (
                            part.tool_name == CodeExecutionTool.kind
                            and code_interpreter_item is not None
                            and isinstance(part.content, dict)
                            and (content := cast(dict[str, Any], part.content))  # pyright: ignore[reportUnknownMemberType]
                            and (status := content.get('status'))
                        ):
                            code_interpreter_item['outputs'] = content.get('outputs')
                            code_interpreter_item['status'] = status
                        elif (
                            part.tool_name == WebSearchTool.kind
                            and web_search_item is not None
                            and isinstance(part.content, dict)  # pyright: ignore[reportUnknownMemberType]
                            and (content := cast(dict[str, Any], part.content))  # pyright: ignore[reportUnknownMemberType]
                            and (status := content.get('status'))
                        ):  # pragma: no branch
                            web_search_item['status'] = status
                elif isinstance(part, ThinkingPart):
                    if part.id and send_item_ids:
                        signature: str | None = None
                        if (
                            part.signature
                            and part.provider_name == system_name
                            and profile_data.openai_supports_encrypted_reasoning_content
                        ):
                            signature = part.signature
                        if (reasoning_item is None or reasoning_item['id'] != part.id) and (signature or part.content):
                            reasoning_item = responses.ResponseReasoningItemParam(
                                id=part.id,
                                summary=[],
                                encrypted_content=signature,
                                type='reasoning',
                            )
                            openai_messages.append(reasoning_item)
                        if part.content:
                            assert reasoning_item is not None
                            reasoning_item['summary'] = [
                                *reasoning_item['summary'],
                                Summary(text=part.content, type='summary_text'),
                            ]
                    else:
                        start_tag, end_tag = profile_data.thinking_tags
                        openai_messages.append(
                            responses.EasyInputMessageParam(
                                role='assistant', content='\n'.join([start_tag, part.content, end_tag])
                            )
                        )
                else:
                    assert_never(part)
        else:
            assert_never(message)
    instructions = get_instructions_fn(messages) or NOT_GIVEN
    return instructions, openai_messages


@overload
async def responses_create(
    model: Any,
    messages: list[ModelMessage],
    stream: Literal[True],
    model_settings: ModelSettings,
    model_request_parameters: ModelRequestParameters,
) -> AsyncStream[responses.ResponseStreamEvent]: ...


@overload
async def responses_create(
    model: Any,
    messages: list[ModelMessage],
    stream: Literal[False],
    model_settings: ModelSettings,
    model_request_parameters: ModelRequestParameters,
) -> responses.Response: ...


async def responses_create(
    model: Any,
    messages: list[ModelMessage],
    stream: bool,
    model_settings: ModelSettings,
    model_request_parameters: ModelRequestParameters,
) -> responses.Response | AsyncStream[responses.ResponseStreamEvent]:
    settings = cast(dict[str, Any], model_settings)
    tools = (
        get_responses_builtin_tools(model.profile, model_request_parameters.builtin_tools)
        + list(settings.get('openai_builtin_tools', []))
        + get_responses_tools(model.profile, list(model_request_parameters.tool_defs.values()))
    )
    if not tools:
        tool_choice: Literal['none', 'required', 'auto'] | None = None
    elif not model_request_parameters.allow_text_output:
        tool_choice = 'required'
    else:
        tool_choice = 'auto'

    previous_response_id = settings.get('openai_previous_response_id')
    messages_to_send = messages
    if previous_response_id == 'auto':
        previous_response_id, messages_to_send = get_responses_previous_response_id_and_new_messages(
            messages, model.system
        )

    instructions, openai_messages = await map_responses_messages(
        model.profile,
        model.system,
        messages_to_send,
        settings,
        map_user_prompt_fn=map_responses_user_prompt,
        get_instructions_fn=model._get_instructions,
    )
    reasoning = get_responses_reasoning(settings)

    text: responses.ResponseTextConfigParam | None = None
    if model_request_parameters.output_mode == 'native':
        output_object = model_request_parameters.output_object
        assert output_object is not None
        text = {'format': map_responses_json_schema(model.profile, output_object)}
    elif model_request_parameters.output_mode == 'prompted' and model.profile.supports_json_object_output:
        text = {'format': {'type': 'json_object'}}
        assert isinstance(instructions, str)
        openai_messages.insert(0, responses.EasyInputMessageParam(role='system', content=instructions))
        instructions = NOT_GIVEN

    if verbosity := settings.get('openai_text_verbosity'):
        text = text or {}
        text['verbosity'] = verbosity

    profile_data = OpenAIModelProfile.from_profile(model.profile)
    unsupported = profile_data.openai_unsupported_model_settings
    for setting in unsupported:
        settings.pop(setting, None)

    include: list[responses.ResponseIncludable] = []
    if profile_data.openai_supports_encrypted_reasoning_content:
        include.append('reasoning.encrypted_content')
    if settings.get('openai_include_code_execution_outputs'):
        include.append('code_interpreter_call.outputs')
    if settings.get('openai_include_web_search_sources'):
        include.append('web_search_call.action.sources')  # type: ignore[arg-type]

    try:
        extra_headers = settings.get('extra_headers', {})
        extra_headers.setdefault('User-Agent', get_user_agent())
        return await model.client.responses.create(
            input=openai_messages,
            model=model._model_name,
            instructions=instructions,
            parallel_tool_calls=settings.get('parallel_tool_calls', NOT_GIVEN),
            tools=tools or NOT_GIVEN,
            tool_choice=tool_choice or NOT_GIVEN,
            max_output_tokens=settings.get('max_tokens', NOT_GIVEN),
            stream=stream,
            temperature=settings.get('temperature', NOT_GIVEN),
            top_p=settings.get('top_p', NOT_GIVEN),
            truncation=settings.get('openai_truncation', NOT_GIVEN),
            timeout=settings.get('timeout', NOT_GIVEN),
            service_tier=settings.get('openai_service_tier', NOT_GIVEN),
            previous_response_id=previous_response_id,
            reasoning=reasoning,
            user=settings.get('openai_user', NOT_GIVEN),
            text=text or NOT_GIVEN,
            include=include or NOT_GIVEN,
            extra_headers=extra_headers,
            extra_body=settings.get('extra_body'),
        )
    except APIStatusError as e:
        if (status_code := e.status_code) >= 400:
            from .. import ModelHTTPError

            raise ModelHTTPError(status_code=status_code, model_name=model.model_name, body=e.body) from e
        raise  # pragma: lax no cover


def process_responses_response(
    model: Any,
    response: responses.Response,
    *,
    map_usage_fn: Callable[[responses.Response], usage.RequestUsage],
    finish_reason_map: Mapping[str | ResponseStatus, FinishReason],
) -> ModelResponse:
    timestamp = number_to_datetime(response.created_at)
    items: list[ModelResponsePart] = []
    for output in response.output:  # pragma: no cover
        if isinstance(output, responses.ResponseReasoningItem):  # pragma: no cover
            signature = output.encrypted_content
            if output.summary:  # pragma: no cover
                for summary in output.summary:  # pragma: no cover
                    items.append(
                        ThinkingPart(
                            content=summary.text,
                            id=output.id,
                            signature=signature,
                            provider_name=model.system if signature else None,
                        )
                    )
                    signature = None
            elif signature:  # pragma: no cover
                items.append(
                    ThinkingPart(
                        content='',
                        id=output.id,
                        signature=signature,
                        provider_name=model.system,
                    )
                )
        elif isinstance(output, responses.ResponseOutputMessage):  # pragma: no cover
            for content in output.content:  # pragma: no cover
                if isinstance(content, responses.ResponseOutputText):  # pragma: no branch
                    items.append(TextPart(content.text, id=output.id))
        elif isinstance(output, responses.ResponseFunctionToolCall):  # pragma: no cover
            items.append(
                ToolCallPart(
                    output.name,
                    output.arguments,
                    tool_call_id=combine_tool_call_ids(output.call_id, output.id),
                )
            )
        elif isinstance(output, responses.ResponseCodeInterpreterToolCall):  # pragma: no cover
            call_part, return_part = map_code_interpreter_tool_call(output, model.system)
            items.extend([call_part, return_part])
        elif isinstance(output, responses.ResponseFunctionWebSearch):  # pragma: no cover
            call_part, return_part = map_web_search_tool_call(output, model.system)
            items.extend([call_part, return_part])
    finish_reason: FinishReason | None = None
    provider_details: dict[str, Any] | None = None
    raw_finish_reason = details.reason if (details := response.incomplete_details) else response.status
    if raw_finish_reason:
        provider_details = {'finish_reason': raw_finish_reason}
        finish_reason = finish_reason_map.get(raw_finish_reason)
    return ModelResponse(
        parts=items,
        usage=map_usage_fn(response),
        model_name=response.model,
        provider_response_id=response.id,
        timestamp=timestamp,
        provider_name=model.system,
        finish_reason=finish_reason,
        provider_details=provider_details,
    )
