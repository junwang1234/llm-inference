from __future__ import annotations

from pathlib import Path

import yaml

from llm_inference.config.schema import ModelConfig, ModelsConfig, ProfileConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_YAML = CONFIG_DIR / "models.yaml"
PROFILES_DIR = CONFIG_DIR / "profiles"


def load_models_config(path: Path = MODELS_YAML) -> ModelsConfig:
    """Load and validate the models.yaml config."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return ModelsConfig(**raw)


def load_profile(profile_name: str) -> ProfileConfig:
    """Load a profile YAML by name (without .yaml extension)."""
    path = PROFILES_DIR / f"{profile_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    return ProfileConfig(**raw["profile"])


def resolve_model(
    model_id: str | None = None,
    profile: str | None = None,
) -> tuple[str, ModelConfig]:
    """Resolve a model ID from explicit arg or profile, returning (id, config).

    Priority: explicit model_id > profile > defaults.model
    """
    config = load_models_config()

    if model_id is None and profile is not None:
        prof = load_profile(profile)
        model_id = prof.model

    if model_id is None:
        model_id = config.defaults.model

    if model_id not in config.models:
        available = ", ".join(config.models.keys())
        raise KeyError(f"Model '{model_id}' not found. Available: {available}")

    return model_id, config.models[model_id]


def get_defaults():
    """Return the defaults section from models.yaml."""
    return load_models_config().defaults
