from __future__ import annotations

from llm_inference.backends.base import InferenceBackend
from llm_inference.config.schema import Defaults, ModelConfig


class VllmBackend(InferenceBackend):
    def compose_env(self, model_id: str, model_config: ModelConfig, defaults: Defaults) -> dict[str, str]:
        return {
            "MODEL_REPO": model_config.hf_repo,
            "MODEL_NAME": model_id,
            "TP_SIZE": str(model_config.gpu_config.tensor_parallel_size),
            "QUANTIZATION": model_config.quantization,
            "DTYPE": model_config.dtype,
            "GPU_MEM_UTIL": str(model_config.gpu_config.gpu_memory_utilization),
            "MAX_MODEL_LEN": str(model_config.context_window),
            "PORT": str(defaults.port),
            "API_KEY": defaults.api_key,
        }

    def compose_service(self) -> str:
        return "vllm"

    def health_check_url(self, port: int) -> str:
        return f"http://localhost:{port}/health"

    def api_base_url(self, port: int) -> str:
        return f"http://localhost:{port}/v1"
