from llm_inference.backends.llamacpp_backend import LlamaCppBackend
from llm_inference.backends.vllm_backend import VllmBackend
from llm_inference.config.loader import load_models_config, get_defaults


def test_vllm_compose_env():
    config = load_models_config()
    defaults = get_defaults()
    model = config.models["qwen2.5-coder-32b-gptq"]
    backend = VllmBackend()

    env = backend.compose_env("qwen2.5-coder-32b-gptq", model, defaults)

    assert env["MODEL_REPO"] == "Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4"
    assert env["MODEL_NAME"] == "qwen2.5-coder-32b-gptq"
    assert env["TP_SIZE"] == "1"
    assert env["QUANTIZATION"] == "gptq"
    assert env["DTYPE"] == "float16"


def test_vllm_service_name():
    assert VllmBackend().compose_service() == "vllm"


def test_llamacpp_compose_env():
    config = load_models_config()
    defaults = get_defaults()
    model = config.models["llama-3.1-70b-q4"]
    backend = LlamaCppBackend()

    env = backend.compose_env("llama-3.1-70b-q4", model, defaults)

    assert env["MODEL_FILE"] == "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf"
    assert env["N_GPU_LAYERS"] == "45"
    assert env["THREADS"] == "16"


def test_llamacpp_service_name():
    assert LlamaCppBackend().compose_service() == "llamacpp"


def test_api_base_urls():
    vllm = VllmBackend()
    llamacpp = LlamaCppBackend()

    assert vllm.api_base_url(8000) == "http://localhost:8000/v1"
    assert llamacpp.api_base_url(8000) == "http://localhost:8000/v1"
