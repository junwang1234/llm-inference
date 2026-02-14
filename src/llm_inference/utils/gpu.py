from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass
class GpuInfo:
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    utilization_pct: int


def detect_gpus() -> list[GpuInfo]:
    """Detect NVIDIA GPUs via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []

    if result.returncode != 0:
        return []

    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            gpus.append(GpuInfo(
                index=int(parts[0]),
                name=parts[1],
                memory_total_mb=int(parts[2]),
                memory_used_mb=int(parts[3]),
                memory_free_mb=int(parts[4]),
                utilization_pct=int(parts[5]),
            ))
    return gpus


def format_gpu_status(gpus: list[GpuInfo]) -> str:
    """Format GPU info for display."""
    if not gpus:
        return "No NVIDIA GPUs detected"

    lines = []
    for gpu in gpus:
        mem_pct = gpu.memory_used_mb / gpu.memory_total_mb * 100
        lines.append(
            f"  GPU {gpu.index}: {gpu.name} | "
            f"{gpu.memory_used_mb}/{gpu.memory_total_mb} MB ({mem_pct:.0f}%) | "
            f"Util: {gpu.utilization_pct}%"
        )
    return "\n".join(lines)
