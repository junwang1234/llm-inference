from __future__ import annotations

import os
import subprocess
from pathlib import Path

from llm_inference.config.loader import PROJECT_ROOT

COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yaml"


def _run(
    args: list[str],
    env: dict[str, str] | None = None,
    capture: bool = False,
    stream: bool = False,
) -> subprocess.CompletedProcess | None:
    """Run a docker compose command."""
    cmd = ["docker", "compose", "-f", str(COMPOSE_FILE)] + args
    run_env = {**os.environ, **(env or {})}

    if stream:
        proc = subprocess.Popen(cmd, env=run_env)
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
        return None

    return subprocess.run(
        cmd,
        env=run_env,
        capture_output=capture,
        text=True,
    )


def compose_up(service: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    """Start a compose service with the given env vars."""
    return _run(
        ["--profile", service, "up", "-d", service],
        env=env,
    )


def compose_down() -> subprocess.CompletedProcess:
    """Stop and remove all compose services."""
    # Need to activate all profiles so `down` can find the services
    return _run(["--profile", "vllm", "--profile", "llamacpp", "down"])


def compose_ps() -> str | None:
    """Return docker compose ps output."""
    result = _run(
        ["--profile", "vllm", "--profile", "llamacpp", "ps"],
        capture=True,
    )
    return result.stdout if result else None


def compose_logs(follow: bool = True, tail: int = 100) -> None:
    """Stream or print compose logs."""
    args = ["--profile", "vllm", "--profile", "llamacpp", "logs", "--tail", str(tail)]
    if follow:
        args.append("-f")
    _run(args, stream=follow, capture=not follow)


def is_container_running() -> bool:
    """Check if the llm-inference container is running."""
    result = subprocess.run(
        ["docker", "ps", "--filter", "name=llm-inference", "--format", "{{.Status}}"],
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def get_container_status() -> dict[str, str]:
    """Get status info about the llm-inference container."""
    result = subprocess.run(
        [
            "docker", "ps", "-a",
            "--filter", "name=llm-inference",
            "--format", "{{.Names}}\t{{.Status}}\t{{.Image}}\t{{.Ports}}",
        ],
        capture_output=True,
        text=True,
    )
    if not result.stdout.strip():
        return {}

    parts = result.stdout.strip().split("\t")
    if len(parts) >= 4:
        return {
            "name": parts[0],
            "status": parts[1],
            "image": parts[2],
            "ports": parts[3],
        }
    return {"raw": result.stdout.strip()}
