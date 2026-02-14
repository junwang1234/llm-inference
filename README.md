# llm-inference

Config-driven local LLM inference server. Manages [vLLM](https://github.com/vllm-project/vllm) and [llama.cpp](https://github.com/ggerganov/llama.cpp) Docker containers with a simple CLI. Add models by editing a YAML file — no code changes needed.

## Prerequisites

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA GPU(s) with CUDA support
- Python 3.11+

## Quick Start

```bash
# Install the CLI
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Start a model
llm-inference serve qwen3-32b-awq

# Test
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-dev-key" \
  -d '{"model":"qwen3-32b-awq","messages":[{"role":"user","content":"Hello"}]}'
```

Or with Docker Compose directly:

```bash
MODEL_REPO="Qwen/Qwen3-32B-AWQ" \
MODEL_NAME="qwen3-32b-awq" \
TP_SIZE=2 \
QUANTIZATION=awq_marlin \
DTYPE=float16 \
GPU_MEM_UTIL=0.90 \
MAX_MODEL_LEN=32768 \
PORT=8000 \
API_KEY="local-dev-key" \
VLLM_EXTRA_ARGS="--enforce-eager --enable-auto-tool-choice --tool-call-parser hermes" \
docker compose --profile vllm up -d vllm
```

## CLI Commands

```
llm-inference serve [MODEL_ID]       # Start a model (stops any running one first)
llm-inference serve --profile NAME   # Start using a named profile
llm-inference stop                   # Stop the running model
llm-inference status                 # Show container status and GPU usage
llm-inference logs                   # Stream container logs
llm-inference list                   # List all configured models
llm-inference download MODEL_ID      # Download model weights from HuggingFace
```

## Configured Models

| Model | Backend | Quantization | VRAM | Use Case |
|-------|---------|-------------|------|----------|
| **Qwen3-32B** | vLLM | AWQ-INT4 | ~22GB (TP=2) | Agentic, reasoning, tool calling |
| Qwen2.5-Coder-32B | vLLM | GPTQ-INT4 | ~20GB (TP=2) | Coding |
| Qwen2.5-Coder-7B | vLLM | AWQ-INT4 | ~5GB | Fast iteration |
| DeepSeek Coder V2 Lite | vLLM | FP16 | ~16GB | Code + reasoning |
| DeepSeek R1 Distill 32B | vLLM | GPTQ-INT4 | ~20GB | Reasoning |
| Llama 3.1 70B | llama.cpp | GGUF Q4 | 48GB + CPU | Max quality |

Add new models by editing `config/models.yaml`.

## Architecture

```
docker-compose.yaml          # vLLM + llama.cpp service definitions
config/
  models.yaml                # Model registry (HF repos, quant, GPU settings)
  profiles/                  # Named profiles for quick switching
src/llm_inference/
  cli.py                     # CLI entry point (click-based)
  backends/
    docker.py                # Docker compose lifecycle management
    vllm_backend.py          # Translates model config to vLLM env vars
    llamacpp_backend.py      # Translates model config to llama.cpp env vars
  config/
    schema.py                # Pydantic validation for YAML configs
    loader.py                # YAML loading and profile merging
  utils/
    gpu.py                   # GPU detection via nvidia-smi
```

Two Docker services (only one runs at a time):
- **vllm** — for quantized models (GPTQ, AWQ). OpenAI-compatible API on port 8000.
- **llamacpp** — for GGUF models needing CPU offloading. Same port.

## OpenAI-Compatible API

The server exposes standard OpenAI endpoints:

- `POST /v1/chat/completions` — chat
- `POST /v1/completions` — text completion
- `GET /v1/models` — list loaded models
- `GET /health` — health check

Works with any OpenAI-compatible client.

## License

MIT
