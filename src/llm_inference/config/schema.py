from __future__ import annotations

from pydantic import BaseModel, Field


class GpuConfig(BaseModel):
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90


class LlamaCppArgs(BaseModel):
    n_gpu_layers: int = 40
    threads: int = 16


class ModelConfig(BaseModel):
    name: str
    hf_repo: str
    backend: str  # "vllm" or "llamacpp"
    quantization: str = "none"
    dtype: str = "auto"
    context_window: int = 32768
    max_tokens: int = 8192
    gpu_config: GpuConfig = Field(default_factory=GpuConfig)
    llamacpp_args: LlamaCppArgs = Field(default_factory=LlamaCppArgs)
    hf_file: str | None = None
    tags: list[str] = Field(default_factory=list)


class Defaults(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    model: str = "qwen2.5-coder-32b-gptq"
    api_key: str = "local-dev-key"


class ModelsConfig(BaseModel):
    models: dict[str, ModelConfig]
    defaults: Defaults = Field(default_factory=Defaults)


class ProfileConfig(BaseModel):
    name: str
    description: str = ""
    model: str
