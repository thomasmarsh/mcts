"""ctypes binding for the `othello-env` cdylib (flat C ABI, numpy arrays in and out).

Build the library first:  cargo build --release -p othello-env
Override the location with the OTHELLO_ENV_LIB environment variable.

A state batch is a C-contiguous `(n, 3)` uint64 array `[black, white, flags]` (see the crate
docs); treat it as opaque. Actions are 0..63 for a square, 64 for PASS.
"""

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

ABI_VERSION = 1
NUM_ACTIONS = 65

_u64 = np.ctypeslib.ndpointer(np.uint64, flags=["C_CONTIGUOUS", "WRITEABLE"])
_u64_in = np.ctypeslib.ndpointer(np.uint64, flags="C_CONTIGUOUS")
_u8_in = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
_u8 = np.ctypeslib.ndpointer(np.uint8, flags=["C_CONTIGUOUS", "WRITEABLE"])
_f32 = np.ctypeslib.ndpointer(np.float32, flags=["C_CONTIGUOUS", "WRITEABLE"])


def _default_lib_path() -> Path:
    suffix = "dylib" if sys.platform == "darwin" else "dll" if sys.platform == "win32" else "so"
    prefix = "" if sys.platform == "win32" else "lib"
    target = Path(__file__).resolve().parents[3] / "target" / "release"
    return target / f"{prefix}othello_env.{suffix}"


def _load() -> ctypes.CDLL:
    lib = ctypes.CDLL(os.environ.get("OTHELLO_ENV_LIB") or str(_default_lib_path()))
    lib.othello_env_abi_version.restype = ctypes.c_uint32
    if lib.othello_env_abi_version() != ABI_VERSION:
        raise RuntimeError("othello_env ABI version mismatch: rebuild the cdylib")
    lib.othello_env_reset_random.argtypes = [ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int32, _u64]
    lib.othello_env_reset_random.restype = None
    lib.othello_env_step.argtypes = [ctypes.c_size_t, _u64, _u8_in, _f32, _u8, ctypes.c_int32]
    lib.othello_env_step.restype = ctypes.c_int64
    lib.othello_env_observe.argtypes = [ctypes.c_size_t, _u64_in, _f32, _u8, ctypes.c_int32]
    lib.othello_env_observe.restype = None
    return lib


_lib = _load()


def reset_random(n: int, max_depth: int, seed: int, parallel: bool = True) -> np.ndarray:
    """`n` non-terminal positions, each after depth ~ U[0, max_depth] uniformly random legal plies."""
    states = np.empty((n, 3), dtype=np.uint64)
    _lib.othello_env_reset_random(n, max_depth, seed, int(parallel), states)
    return states


def step(states: np.ndarray, actions: np.ndarray, parallel: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Advance every env in place. Returns `(reward, done)`; reward (+1/0/-1) is for the player who
    just moved. Raises ValueError, leaving the offending envs unchanged, on an illegal action or an
    env that is already terminal."""
    n = len(states)
    actions = np.ascontiguousarray(actions, dtype=np.uint8)
    reward = np.empty(n, dtype=np.float32)
    done = np.empty(n, dtype=np.uint8)
    rejected = _lib.othello_env_step(n, states, actions, reward, done, int(parallel))
    if rejected:
        raise ValueError(f"{rejected} of {n} envs got an illegal action or were already terminal")
    return reward, done.view(np.bool_)


def observe(states: np.ndarray, parallel: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Mover-relative `(n, 2, 8, 8)` float32 planes (own, opponent) and an `(n, 65)` bool legal mask."""
    n = len(states)
    obs = np.empty((n, 2, 8, 8), dtype=np.float32)
    mask = np.empty((n, NUM_ACTIONS), dtype=np.uint8)
    _lib.othello_env_observe(n, states, obs, mask, int(parallel))
    return obs, mask.view(np.bool_)
