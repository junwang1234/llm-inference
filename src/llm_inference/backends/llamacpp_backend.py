from __future__ import annotations

from llm_inference.backends.base import InferenceBackend
from llm_inference.config.schema import Defaults, ModelConfig


class LlamaCppBackend(InferenceBackend):
    def compose_env(self, model_id: str, model_config: ModelConfig, defaults: Defaults) -> dict[str, str]:
        if model_config.hf_file is None:
            raise ValueError(f"Model '{model_id}' requires 'hf_file' for llama.cpp backend")

        # llama.cpp expects the model file under /models/ in the container
        # The HF cache is mounted at /models, so we need the path relative to the
        # hub cache structure: hub/models--<org>--<repo>/snapshots/<hash>/<file>
        # For simplicity, we pass just the filename and expect the user to have
        # downloaded it. The download command places it correctly.
        return {
            "MODEL_FILE": model_config.hf_file,
            "N_GPU_LAYERS": str(model_config.llamacpp_args.n_gpu_layers),
            "THREADS": str(model_config.llamacpp_args.threads),
            "CTX_SIZE": str(model_config.context_window),
            "PORT": str(defaults.port),
        }

    def compose_service(self) -> str:
        return "llamacpp"

    def health_check_url(self, port: int) -> str:
        return f"http://localhost:{port}/health"

    def api_base_url(self, port: int) -> str:
        return f"http://localhost:{port}/v1"
