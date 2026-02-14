from __future__ import annotations

import subprocess
import sys

import click
from rich.console import Console
from rich.table import Table

from llm_inference.backends import docker
from llm_inference.backends.llamacpp_backend import LlamaCppBackend
from llm_inference.backends.vllm_backend import VllmBackend
from llm_inference.config.loader import get_defaults, load_models_config, resolve_model
from llm_inference.utils.gpu import detect_gpus, format_gpu_status

console = Console()

BACKENDS = {
    "vllm": VllmBackend(),
    "llamacpp": LlamaCppBackend(),
}


@click.group()
def cli():
    """Local LLM inference server management."""


@cli.command()
@click.argument("model_id", required=False)
@click.option("--profile", "-p", help="Use a named profile (e.g. coding-fast)")
def serve(model_id: str | None, profile: str | None):
    """Start serving a model. Stops any running model first."""
    try:
        model_id, model_config = resolve_model(model_id, profile)
    except (KeyError, FileNotFoundError) as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    defaults = get_defaults()
    backend = BACKENDS.get(model_config.backend)
    if backend is None:
        console.print(f"[red]Error:[/red] Unknown backend '{model_config.backend}'")
        sys.exit(1)

    # Stop any running container first
    if docker.is_container_running():
        console.print("[yellow]Stopping existing model...[/yellow]")
        docker.compose_down()

    console.print(f"[green]Starting:[/green] {model_config.name}")
    console.print(f"  Model:   {model_config.hf_repo}")
    console.print(f"  Backend: {model_config.backend}")
    console.print(f"  Port:    {defaults.port}")

    env = backend.compose_env(model_id, model_config, defaults)
    service = backend.compose_service()

    result = docker.compose_up(service, env)
    if result and result.returncode != 0:
        console.print("[red]Failed to start container.[/red]")
        sys.exit(1)

    console.print(f"\n[green]Model is starting.[/green] Check status with: llm-inference status")
    console.print(f"API will be available at: http://localhost:{defaults.port}/v1")


@cli.command()
def stop():
    """Stop the running model."""
    if not docker.is_container_running():
        console.print("[yellow]No model is running.[/yellow]")
        return

    console.print("[yellow]Stopping model...[/yellow]")
    docker.compose_down()
    console.print("[green]Stopped.[/green]")


@cli.command()
def status():
    """Show status of the running model and GPU usage."""
    container = docker.get_container_status()
    if not container:
        console.print("[yellow]No model container found.[/yellow]")
    else:
        console.print("[bold]Container:[/bold]")
        console.print(f"  Name:   {container.get('name', 'N/A')}")
        console.print(f"  Status: {container.get('status', 'N/A')}")
        console.print(f"  Image:  {container.get('image', 'N/A')}")
        console.print(f"  Ports:  {container.get('ports', 'N/A')}")

    console.print("\n[bold]GPUs:[/bold]")
    gpus = detect_gpus()
    console.print(format_gpu_status(gpus))


@cli.command()
@click.option("--follow/--no-follow", "-f", default=True, help="Follow log output")
@click.option("--tail", "-n", default=100, help="Number of lines to show")
def logs(follow: bool, tail: int):
    """Show container logs."""
    if not docker.is_container_running():
        console.print("[yellow]No model is running.[/yellow]")
        return
    docker.compose_logs(follow=follow, tail=tail)


@cli.command("list")
def list_models():
    """List all configured models."""
    config = load_models_config()
    table = Table(title="Configured Models")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Backend", style="green")
    table.add_column("Quant")
    table.add_column("Context")
    table.add_column("Tags")

    default_model = config.defaults.model
    for model_id, model in config.models.items():
        marker = " *" if model_id == default_model else ""
        table.add_row(
            model_id + marker,
            model.name,
            model.backend,
            model.quantization,
            str(model.context_window),
            ", ".join(model.tags),
        )

    console.print(table)
    console.print(f"\n[dim]* = default model[/dim]")


@cli.command()
@click.argument("model_id")
def download(model_id: str):
    """Download model weights from HuggingFace."""
    try:
        model_id_resolved, model_config = resolve_model(model_id)
    except KeyError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    console.print(f"[green]Downloading:[/green] {model_config.name}")
    console.print(f"  Repo: {model_config.hf_repo}")

    if model_config.hf_file:
        # Download specific GGUF file
        cmd = [
            "huggingface-cli", "download",
            model_config.hf_repo,
            model_config.hf_file,
        ]
    else:
        # Download full repo
        cmd = [
            "huggingface-cli", "download",
            model_config.hf_repo,
        ]

    console.print(f"  Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        console.print("[red]Download failed.[/red]")
        sys.exit(1)
    console.print("[green]Download complete.[/green]")
