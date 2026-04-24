"""Helpers for configuring the CUDA runtime for repo-local GPU scripts."""

from __future__ import annotations

import os
import site
from pathlib import Path


def _find_site_packages_root() -> Path:
    for candidate in site.getsitepackages():
        candidate_path = Path(candidate)
        if candidate_path.name == "site-packages":
            return candidate_path
    raise RuntimeError("Could not locate a Python site-packages directory.")


def configure_cuda_runtime() -> list[Path]:
    """Configure DLL discovery so CuPy can use the packaged CUDA libraries."""

    site_packages_root = _find_site_packages_root()
    runtime_root = site_packages_root / "nvidia" / "cuda_runtime"
    runtime_bin = runtime_root / "bin"
    nvrtc_bin = site_packages_root / "nvidia" / "cuda_nvrtc" / "bin"
    nvjitlink_bin = site_packages_root / "nvidia" / "nvjitlink" / "bin"

    required_paths = [runtime_bin, nvrtc_bin, nvjitlink_bin]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        missing_list = ", ".join(str(path) for path in missing_paths)
        raise RuntimeError(
            "The packaged CUDA runtime is incomplete. Missing paths: "
            f"{missing_list}"
        )

    os.environ["CUDA_PATH"] = str(runtime_root)

    existing_path_parts = [
        path_part
        for path_part in os.environ.get("PATH", "").split(os.pathsep)
        if path_part
    ]
    for dll_path in required_paths:
        dll_path_str = str(dll_path)
        if dll_path_str not in existing_path_parts:
            existing_path_parts.insert(0, dll_path_str)
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(dll_path_str)
    os.environ["PATH"] = os.pathsep.join(existing_path_parts)

    return required_paths


__all__ = ["configure_cuda_runtime"]
