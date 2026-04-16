from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import torch


def get_clip_module():
    try:
        import clip as clip_module

        return clip_module
    except ModuleNotFoundError:
        fallback_root = Path(__file__).resolve().parents[1] / "_tmp_tda_repo"
        if fallback_root.exists() and str(fallback_root) not in sys.path:
            sys.path.insert(0, str(fallback_root))
            try:
                import ftfy  # noqa: F401
            except ModuleNotFoundError:
                shim = types.ModuleType("ftfy")
                shim.fix_text = lambda text: text
                sys.modules["ftfy"] = shim
            import clip as clip_module

            return clip_module
        raise


def get_extraction_runtime(
    *,
    default_cuda_batch_size: int = 32,
    default_cpu_batch_size: int = 64,
    default_num_workers: int = 4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size_env = os.getenv("AIP_EXTRACT_BATCH_SIZE")
    num_workers_env = os.getenv("AIP_EXTRACT_NUM_WORKERS")

    if batch_size_env is not None:
        batch_size = max(1, int(batch_size_env))
    else:
        batch_size = default_cuda_batch_size if device.type == "cuda" else default_cpu_batch_size

    if num_workers_env is not None:
        num_workers = max(0, int(num_workers_env))
    else:
        num_workers = default_num_workers

    pin_memory = device.type == "cuda"
    return device, batch_size, num_workers, pin_memory
