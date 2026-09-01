from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BRIDGE_PATH = ROOT / "native" / "bin" / "bokujuu_dlss_sr_bridge.dll"
PROJECT_ID = "8d6c9ee1-2515-49f3-a30c-bb310d38f9bc"


class DLSSRuntimeError(RuntimeError):
    pass


def _models_dir() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.models_dir)
    except ImportError:
        return ROOT.parents[1] / "models"


def runtime_candidates() -> list[Path]:
    candidates = []
    configured = os.environ.get("BOKUJUU_DLSS_RUNTIME")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            _models_dir() / "dlss5" / "nvngx_dlss.dll",
            _models_dir() / "dlss" / "nvngx_dlss.dll",
        ]
    )
    return candidates


def resolve_runtime(path: str | os.PathLike | None = None) -> Path:
    candidates = [Path(path)] if path else runtime_candidates()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file() and candidate.stat().st_size >= 1024 * 1024:
            return candidate
    expected = "\n".join(f"- {candidate}" for candidate in candidates)
    raise DLSSRuntimeError(
        "DLSS Super Resolution runtime was not found.\n"
        f"Expected one of:\n{expected}\n"
        "Place a legally obtained NVIDIA nvngx_dlss.dll in ComfyUI/models/dlss5/."
    )


def resolve_ngx_core() -> Path:
    configured = os.environ.get("BOKUJUU_NGX_CORE")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if candidate.is_file():
            return candidate

    driver_store = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "DriverStore" / "FileRepository"
    candidates = sorted(driver_store.glob("nv*_dispi.inf_*/*nvngx.dll"), reverse=True)
    if not candidates:
        candidates = sorted(driver_store.glob("**/nvngx.dll"), reverse=True)
    if candidates:
        return candidates[0]
    raise DLSSRuntimeError(
        "NVIDIA NGX core (nvngx.dll) was not found in the installed display driver. "
        "Install a current NVIDIA driver."
    )


def resolve_bridge() -> Path:
    if BRIDGE_PATH.is_file():
        return BRIDGE_PATH
    raise DLSSRuntimeError(
        f"Bokujuu DLSS bridge was not built: {BRIDGE_PATH}\n"
        "Run native/build_bridge.ps1 with a local NVIDIA DLSS SDK checkout."
    )


class NativeDLSSSession:
    _ERROR_CAPACITY = 4096

    def __init__(
        self,
        input_width: int,
        input_height: int,
        output_width: int,
        output_height: int,
        quality: int,
        gpu_index: int = 0,
        runtime_path: str | os.PathLike | None = None,
    ):
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.output_width = int(output_width)
        self.output_height = int(output_height)
        self._library = ctypes.CDLL(str(resolve_bridge()))
        self._configure_abi()
        error = ctypes.create_string_buffer(self._ERROR_CAPACITY)
        self._handle = self._library.bokujuu_dlss_create(
            str(resolve_runtime(runtime_path)),
            str(resolve_ngx_core()),
            PROJECT_ID.encode("ascii"),
            self.input_width,
            self.input_height,
            self.output_width,
            self.output_height,
            int(quality),
            int(gpu_index),
            error,
            self._ERROR_CAPACITY,
        )
        if not self._handle:
            raise DLSSRuntimeError(error.value.decode("utf-8", errors="replace"))

    def _configure_abi(self):
        library = self._library
        library.bokujuu_dlss_create.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        library.bokujuu_dlss_create.restype = ctypes.c_void_p
        library.bokujuu_dlss_process.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        library.bokujuu_dlss_process.restype = ctypes.c_int
        library.bokujuu_dlss_destroy.argtypes = [ctypes.c_void_p]
        library.bokujuu_dlss_destroy.restype = None

    def process(self, rgba: np.ndarray, depth: np.ndarray, reset: bool) -> np.ndarray:
        rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
        depth = np.ascontiguousarray(depth, dtype=np.float32)
        if rgba.shape != (self.input_height, self.input_width, 4):
            raise ValueError(f"Expected RGBA shape {(self.input_height, self.input_width, 4)}; got {rgba.shape}")
        if depth.shape != (self.input_height, self.input_width):
            raise ValueError(f"Expected depth shape {(self.input_height, self.input_width)}; got {depth.shape}")
        output = np.empty((self.output_height, self.output_width, 4), dtype=np.uint8)
        error = ctypes.create_string_buffer(self._ERROR_CAPACITY)
        ok = self._library.bokujuu_dlss_process(
            self._handle,
            rgba.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            rgba.strides[0],
            depth.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            depth.strides[0],
            int(bool(reset)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            output.strides[0],
            error,
            self._ERROR_CAPACITY,
        )
        if not ok:
            raise DLSSRuntimeError(error.value.decode("utf-8", errors="replace"))
        return output

    def close(self):
        if getattr(self, "_handle", None):
            self._library.bokujuu_dlss_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
