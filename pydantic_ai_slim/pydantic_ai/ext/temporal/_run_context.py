from __future__ import annotations

from typing import Any, Callable

from pydantic_ai._run_context import RunContext


class TemporalRunContext(RunContext[Any]):
    def __init__(self, **kwargs: Any):
        self.__dict__ = kwargs
        setattr(
            self,
            '__dataclass_fields__',
            {name: field for name, field in RunContext.__dataclass_fields__.items() if name in kwargs},
        )

    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError as e:
            if name in RunContext.__dataclass_fields__:
                raise AttributeError(
                    f'Temporalized {RunContext.__name__!r} object has no attribute {name!r}. '
                    'To make the attribute available, pass a `TemporalSettings` object to `temporalize_agent` with a custom `serialize_run_context` function that returns a dictionary that includes the attribute.'
                )
            else:
                raise e

    @classmethod
    def serialize_run_context(
        cls,
        ctx: RunContext[Any],
        extra_serializer: Callable[[RunContext[Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {
            'retries': ctx.retries,
            'tool_call_id': ctx.tool_call_id,
            'tool_name': ctx.tool_name,
            'retry': ctx.retry,
            'run_step': ctx.run_step,
            **(extra_serializer(ctx) if extra_serializer else {}),
        }

    @classmethod
    def deserialize_run_context(
        cls, ctx: dict[str, Any], extra_deserializer: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    ) -> RunContext[Any]:
        return cls(
            retries=ctx['retries'],
            tool_call_id=ctx['tool_call_id'],
            tool_name=ctx['tool_name'],
            retry=ctx['retry'],
            run_step=ctx['run_step'],
            **(extra_deserializer(ctx) if extra_deserializer else {}),
        )


def serialize_run_context_deps(ctx: RunContext[Any]) -> dict[str, Any]:
    if not isinstance(ctx.deps, dict):
        raise ValueError(
            'The `deps` object must be a JSON-serializable dictionary in order to be used with Temporal. '
            'To use a different type, pass a `TemporalSettings` object to `temporalize_agent` with custom `serialize_run_context` and `deserialize_run_context` functions.'
        )
    return {'deps': ctx.deps}  # pyright: ignore[reportUnknownMemberType]


def deserialize_run_context_deps(ctx: dict[str, Any]) -> dict[str, Any]:
    return {'deps': ctx['deps']}
