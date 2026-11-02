"""Torch-free inference for Stable-Baselines3 PPO MlpPolicy models.

Reads ``policy.pth`` from an SB3 ``.zip`` with a restricted unpickler and runs the
deterministic forward pass in numpy, so low-memory hosts avoid importing torch.
"""

from __future__ import annotations

import json
import pickle
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

_STORAGE_DTYPES = {
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "HalfStorage": np.float16,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
}


class _StorageType:
    def __init__(self, name: str) -> None:
        self.dtype = np.dtype(_STORAGE_DTYPES[name])


def _rebuild_tensor(storage, offset, size, stride, *_args):
    size = tuple(int(s) for s in size)
    stride = tuple(int(s) * storage.itemsize for s in stride)
    return np.lib.stride_tricks.as_strided(storage[int(offset):], shape=size, strides=stride).copy()


def _rebuild_parameter(data, *_args):
    return data


class _Unpickler(pickle.Unpickler):
    def __init__(self, handle, archive: zipfile.ZipFile, prefix: str) -> None:
        super().__init__(handle)
        self._archive = archive
        self._prefix = prefix

    def find_class(self, module: str, name: str):
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "torch" and name in _STORAGE_DTYPES:
            return _StorageType(name)
        raise pickle.UnpicklingError(f"unsupported global {module}.{name}")

    def persistent_load(self, pid):
        _tag, storage_type, key, _location, _numel = pid
        raw = self._archive.read(f"{self._prefix}/data/{key}")
        return np.frombuffer(raw, dtype=storage_type.dtype)


def _load_state_dict(archive: zipfile.ZipFile) -> "OrderedDict[str, np.ndarray]":
    import io

    pkl_name = next(n for n in archive.namelist() if n.endswith("data.pkl"))
    prefix = pkl_name[: -len("/data.pkl")]
    return _Unpickler(io.BytesIO(archive.read(pkl_name)), archive, prefix).load()


class NumpyPPOPolicy:
    """Deterministic MlpPolicy (tanh, separate pi/vf nets) with a Box action space."""

    def __init__(self, model_path: Path) -> None:
        with zipfile.ZipFile(model_path) as outer:
            data = json.loads(outer.read("data"))
            with outer.open("policy.pth") as handle:
                import io

                with zipfile.ZipFile(io.BytesIO(handle.read())) as inner:
                    state = _load_state_dict(inner)

        kwargs = data.get("policy_kwargs") or {}
        if kwargs.get("net_arch") not in (None, {}) or "activation_fn" in kwargs:
            raise ValueError("custom policy architecture is not supported by numpy loader")

        self._layers = []
        index = 0
        while f"mlp_extractor.policy_net.{index}.weight" in state:
            self._layers.append(
                (
                    state[f"mlp_extractor.policy_net.{index}.weight"].astype(np.float32),
                    state[f"mlp_extractor.policy_net.{index}.bias"].astype(np.float32),
                )
            )
            index += 2  # Sequential(Linear, Tanh, Linear, Tanh, ...)
        if not self._layers:
            raise ValueError("no policy_net layers found in model")
        self._action_w = state["action_net.weight"].astype(np.float32)
        self._action_b = state["action_net.bias"].astype(np.float32)

        space = data["observation_space"]
        self.observation_space = _Space(tuple(space["_shape"]))
        action_space = data["action_space"]
        self._action_low = np.array(_parse_bound(action_space["low"]), dtype=np.float32)
        self._action_high = np.array(_parse_bound(action_space["high"]), dtype=np.float32)

    def predict(self, observation, deterministic: bool = True) -> Tuple[np.ndarray, Optional[Any]]:
        x = np.asarray(observation, dtype=np.float32).reshape(-1)
        for weight, bias in self._layers:
            x = np.tanh(weight @ x + bias)
        action = self._action_w @ x + self._action_b
        return np.clip(action, self._action_low, self._action_high), None


class _Space:
    def __init__(self, shape: tuple) -> None:
        self.shape = shape


def _parse_bound(text: str):
    return [float(v) for v in text.strip("[]").split()]


def load_numpy_policy(model_path: Path) -> Optional[NumpyPPOPolicy]:
    """Return a torch-free policy, or None if the model cannot be read this way."""
    try:
        return NumpyPPOPolicy(Path(model_path))
    except Exception:
        return None
