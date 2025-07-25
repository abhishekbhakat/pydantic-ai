from __future__ import annotations

from dataclasses import dataclass, fields, replace
from datetime import timedelta
from typing import Any, Callable

from temporalio.common import Priority, RetryPolicy
from temporalio.workflow import ActivityCancellationType, VersioningIntent

from pydantic_ai._run_context import RunContext

from ._run_context import deserialize_run_context_deps, serialize_run_context_deps


@dataclass
class TemporalSettings:
    """Settings for Temporal `execute_activity` and Pydantic AI-specific Temporal activity behavior."""

    # Temporal settings
    task_queue: str | None = None
    schedule_to_close_timeout: timedelta | None = None
    schedule_to_start_timeout: timedelta | None = None
    start_to_close_timeout: timedelta | None = None
    heartbeat_timeout: timedelta | None = None
    retry_policy: RetryPolicy | None = None
    cancellation_type: ActivityCancellationType = ActivityCancellationType.TRY_CANCEL
    activity_id: str | None = None
    versioning_intent: VersioningIntent | None = None
    summary: str | None = None
    priority: Priority = Priority.default

    serialize_run_context: Callable[[RunContext], dict[str, Any]] = serialize_run_context_deps
    deserialize_run_context: Callable[[dict[str, Any]], dict[str, Any]] = deserialize_run_context_deps

    @property
    def execute_activity_options(self) -> dict[str, Any]:
        return {
            'task_queue': self.task_queue,
            'schedule_to_close_timeout': self.schedule_to_close_timeout,
            'schedule_to_start_timeout': self.schedule_to_start_timeout,
            'start_to_close_timeout': self.start_to_close_timeout,
            'heartbeat_timeout': self.heartbeat_timeout,
            'retry_policy': self.retry_policy,
            'cancellation_type': self.cancellation_type,
            'activity_id': self.activity_id,
            'versioning_intent': self.versioning_intent,
            'summary': self.summary,
            'priority': self.priority,
        }

    def merge(self, other: TemporalSettings | None) -> TemporalSettings:
        """Merge non-default values from another TemporalSettings instance into this one, returning a new instance."""
        if not other:
            return self
        return replace(self, **{f.name: value for f in fields(other) if (value := getattr(other, f.name)) != f.default})
