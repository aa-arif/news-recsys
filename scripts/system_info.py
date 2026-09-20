"""Record the hardware and library versions every latency number was measured on.

A p99 without hardware attached is not a result. This writes the machine description
once, and the load-test report embeds it, so the numbers in the README cannot drift away
from the box that produced them.
"""

from __future__ import annotations

import argparse
import platform
import sys
from typing import Any

from news_recsys.config import get_settings
from news_recsys.io_utils import write_json
from news_recsys.logging_utils import get_logger

logger = get_logger("scripts.system_info")


def cpu_name() -> str:
    if platform.system() == "Windows":
        import subprocess

        try:
            output = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if output.returncode == 0 and output.stdout.strip():
                return output.stdout.strip()
        except Exception:  # pragma: no cover - best effort only
            pass
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:  # pragma: no cover
            pass
    return platform.processor() or platform.machine()


def memory_gb() -> float:
    try:
        import psutil

        return round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:  # pragma: no cover - psutil is optional at runtime
        return float("nan")


def library_versions() -> dict[str, str]:
    import faiss
    import lightgbm
    import numpy
    import onnxruntime
    import torch

    return {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "onnxruntime": onnxruntime.__version__,
        "faiss": faiss.__version__,
        "lightgbm": lightgbm.__version__,
    }


def collect() -> dict[str, Any]:
    import os

    import psutil
    import torch

    return {
        "cpu": cpu_name(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": os.cpu_count(),
        "memory_gb": memory_gb(),
        "platform": platform.platform(),
        "torch_threads_default": torch.get_num_threads(),
        "gpu": "none (CPU-only measurements)",
        "libraries": library_versions(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, choices=["small", "large", "synthetic"])
    args = parser.parse_args()
    settings = get_settings(dataset=args.dataset) if args.dataset else get_settings()
    settings.ensure_dirs()

    info = collect()
    path = write_json(settings.metrics_dir / "system_info.json", info)
    logger.info(
        "%s | %s physical cores / %s logical | %.1f GB RAM",
        info["cpu"],
        info["physical_cores"],
        info["logical_cores"],
        info["memory_gb"],
    )
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
