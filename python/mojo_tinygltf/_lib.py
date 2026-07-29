"""ctypes loader for the compiled Mojo kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "tinygltf.mojo")
LIB = os.environ.get("MOJO_TINYGLTF_LIB") or os.path.join(
    ROOT, "dist", "libmojo-tinygltf.so"
)

I = ctypes.c_int64

_SIGNATURES = {
    "mtg_byte_stride": ([I, I, I], I),
    "mtg_decode_accessor": ([I] * 12, I),
    "mtg_apply_sparse": ([I] * 15, I),
    "mtg_local_matrices": ([I] * 9, I),
    "mtg_resolve_world": ([I] * 9, I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_TINYGLTF_LIB") and os.path.exists(LIB) and not force:
        return LIB
    if not force and os.path.exists(LIB):
        if os.path.getmtime(LIB) >= os.path.getmtime(SRC):
            return LIB
    pixi = shutil.which("pixi")
    if not pixi:
        raise BuildError("pixi not found; run `pixi run build` first")
    proc = subprocess.run(
        [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "build"],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_instance: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _instance
    if _instance is None:
        _instance = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_instance, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _instance


def addr(array) -> int:
    return int(array.ctypes.data)
