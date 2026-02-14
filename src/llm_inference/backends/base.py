from __future__ import annotations

from abc import ABC, abstractmethod

from llm_inference.config.schema import Defaults, ModelConfig


class InferenceBackend(ABC):
    @abstractmethod
    def compose_env(self, model_id: str, model_config: ModelConfig, defaults: Defaults) -> dict[str, str]:
        """Return environment variables for docker compose."""

    @abstractmethod
    def compose_service(self) -> str:
        """Return the docker compose service name ('vllm' or 'llamacpp')."""

    @abstractmethod
    def health_check_url(self, port: int) -> str:
        """Return the health check URL."""

    @abstractmethod
    def api_base_url(self, port: int) -> str:
        """Return the OpenAI-compatible API base URL."""
