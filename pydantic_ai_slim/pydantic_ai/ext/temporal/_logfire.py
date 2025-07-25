from __future__ import annotations

from typing import Callable

from opentelemetry.trace import get_tracer
from temporalio.client import ClientConfig, Plugin as ClientPlugin
from temporalio.contrib.opentelemetry import TracingInterceptor
from temporalio.runtime import OpenTelemetryConfig, Runtime, TelemetryConfig
from temporalio.service import ConnectConfig, ServiceClient


def _default_setup_logfire():
    import logfire

    logfire.configure(console=False)
    logfire.instrument_pydantic_ai()


class LogfirePlugin(ClientPlugin):
    """Temporal client plugin for Logfire."""

    def __init__(self, setup_logfire: Callable[[], None] = _default_setup_logfire):
        self.setup_logfire = setup_logfire

    def configure_client(self, config: ClientConfig) -> ClientConfig:
        interceptors = config.get('interceptors', [])
        config['interceptors'] = [*interceptors, TracingInterceptor(get_tracer('temporal'))]
        return super().configure_client(config)

    async def connect_service_client(self, config: ConnectConfig) -> ServiceClient:
        self.setup_logfire()

        config.runtime = Runtime(telemetry=TelemetryConfig(metrics=OpenTelemetryConfig(url='http://localhost:4318')))
        return await super().connect_service_client(config)
