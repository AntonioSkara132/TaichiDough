"""Explicit Taichi runtime selection; availability is not execution evidence."""
import os
import platform

import taichi as ti
from taichi.lang import impl


_RUNTIME_RECORD = None


def init_runtime(backend="cpu", precision="f32", cpu_threads=1, debug=False, seed=0):
    global _RUNTIME_RECORD
    architectures = {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}
    if backend not in architectures or precision not in {"f32", "f64"}:
        raise ValueError("Require backend cpu/cuda/vulkan and precision f32/f64")
    if backend == "vulkan" and precision == "f64":
        raise ValueError("This experiment qualifies Vulkan in f32 only; use CPU for f64 tests")
    if not isinstance(cpu_threads, int) or cpu_threads < 1:
        raise ValueError("cpu_threads must be a positive integer")
    requested = {"backend": backend, "precision": precision, "cpu_threads": cpu_threads,
                 "debug": bool(debug), "seed": int(seed)}
    if _RUNTIME_RECORD is not None:
        if any(_RUNTIME_RECORD[k] != v for k, v in requested.items()):
            raise RuntimeError("Taichi is already initialized differently; use a separate process")
        return dict(_RUNTIME_RECORD)
    ti.init(arch=architectures[backend], default_fp=ti.f32 if precision == "f32" else ti.f64,
            cpu_max_num_threads=cpu_threads, debug=debug, random_seed=int(seed),
            enable_fallback=False, offline_cache=False)
    actual = impl.current_cfg().arch
    if actual != architectures[backend]:
        raise RuntimeError(f"Requested {backend}, but Taichi selected {actual}; fallback is not accepted")
    _RUNTIME_RECORD = {
        **requested, "actual_arch": str(actual), "taichi_version": ti.__version__,
        "host": platform.node(), "platform": platform.platform(),
        "machine": platform.machine(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_identity": "See Taichi initialization log; architecture alone does not identify a GPU device.",
        "initialization_verified": True, "forward_verified": False, "backward_verified": False,
    }
    return dict(_RUNTIME_RECORD)
