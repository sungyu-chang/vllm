import torch

from vllm.profiler import scopes


def test_moe_profile_scope_tracks_layer_for_comm_when_enabled(monkeypatch):
    monkeypatch.setenv("VLLM_CUSTOM_SCOPES_FOR_PROFILING", "1")
    monkeypatch.setenv("VLLM_PROFILE_LAYER_SCOPES", "1")
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)

    with scopes.moe_profile_scope("vllm:fused_moe", "model.layers.0.mlp.experts"):
        assert scopes.moe_comm_detail("dispatch") == (
            "model.layers.0.mlp.experts:dispatch"
        )

    assert scopes.moe_comm_detail("dispatch") == "dispatch"


def test_moe_profile_scope_skips_layer_context_when_disabled(monkeypatch):
    monkeypatch.setenv("VLLM_CUSTOM_SCOPES_FOR_PROFILING", "1")
    monkeypatch.setenv("VLLM_PROFILE_LAYER_SCOPES", "0")
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)

    with scopes.moe_profile_scope("vllm:fused_moe", "model.layers.0.mlp.experts"):
        assert scopes.moe_comm_detail("dispatch") == "dispatch"


def test_moe_profile_scope_skips_layer_context_while_compiling(monkeypatch):
    monkeypatch.setenv("VLLM_CUSTOM_SCOPES_FOR_PROFILING", "1")
    monkeypatch.setenv("VLLM_PROFILE_LAYER_SCOPES", "1")
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    with scopes.moe_profile_scope("vllm:fused_moe", "model.layers.0.mlp.experts"):
        assert scopes.moe_comm_detail("dispatch") == "dispatch"