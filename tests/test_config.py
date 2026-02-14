from llm_inference.config.loader import load_models_config, resolve_model
from llm_inference.config.schema import ModelsConfig


def test_load_models_config():
    config = load_models_config()
    assert isinstance(config, ModelsConfig)
    assert len(config.models) > 0
    assert "qwen2.5-coder-32b-gptq" in config.models


def test_resolve_model_explicit():
    model_id, model_config = resolve_model("qwen2.5-coder-7b-awq")
    assert model_id == "qwen2.5-coder-7b-awq"
    assert model_config.backend == "vllm"
    assert model_config.quantization == "awq"


def test_resolve_model_default():
    model_id, model_config = resolve_model()
    assert model_id == "qwen2.5-coder-32b-gptq"


def test_resolve_model_profile():
    model_id, model_config = resolve_model(profile="coding-fast")
    assert model_id == "qwen2.5-coder-7b-awq"


def test_model_config_fields():
    config = load_models_config()
    model = config.models["qwen2.5-coder-32b-gptq"]
    assert model.gpu_config.tensor_parallel_size == 1
    assert model.gpu_config.gpu_memory_utilization == 0.90
    assert model.context_window == 32768
    assert "coding" in model.tags


def test_llamacpp_model():
    config = load_models_config()
    model = config.models["llama-3.1-70b-q4"]
    assert model.backend == "llamacpp"
    assert model.hf_file is not None
    assert model.llamacpp_args.n_gpu_layers == 45
