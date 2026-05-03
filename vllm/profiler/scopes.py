# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional profiler scopes for coarse model-module timing."""

import os
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar

from torch.autograd.profiler import record_function

import vllm.envs as envs

_CURRENT_MOE_LAYER: ContextVar[str | None] = ContextVar(
    "current_moe_layer",
    default=None,
)


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in ("1", "true", "yes", "on")


@contextmanager
def _nested_profile_scope(name: str, detail: str) -> Iterator[None]:
    with record_function(name):
        with record_function(f"{name}:{detail}"):
            yield


def module_profile_scope(
    name: str,
    detail: str | None = None,
) -> AbstractContextManager:
    if envs.VLLM_CUSTOM_SCOPES_FOR_PROFILING:
        if detail is not None and _env_enabled("VLLM_PROFILE_LAYER_SCOPES"):
            return _nested_profile_scope(name, detail)
        return record_function(name)
    return nullcontext()


@contextmanager
def moe_profile_scope(name: str, layer_name: str) -> Iterator[None]:
    token = _CURRENT_MOE_LAYER.set(layer_name)
    try:
        with module_profile_scope(name, layer_name):
            yield
    finally:
        _CURRENT_MOE_LAYER.reset(token)


def moe_comm_detail(name: str) -> str:
    layer_name = _CURRENT_MOE_LAYER.get()
    if layer_name is None:
        return name
    return f"{layer_name}:{name}"
