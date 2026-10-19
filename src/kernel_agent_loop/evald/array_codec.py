"""Encode numpy arrays for JSONL IPC between the eval parent and runner child."""

from __future__ import annotations

import base64

import numpy as np


def encode_array(array: np.ndarray) -> dict:
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def decode_array(payload: dict) -> np.ndarray:
    return np.frombuffer(base64.b64decode(payload["data"]), dtype=np.dtype(payload["dtype"])).reshape(
        payload["shape"]
    ).copy()
