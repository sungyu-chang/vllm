# SPDX-License-Identifier: Apache-2.0
"""Load MoE hyperparameters from HuggingFace model configs or built-in presets."""

from dataclasses import dataclass


@dataclass
class MoEModelConfig:
    num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    model_name: str = "custom"


# Built-in presets (no network required)
PRESETS: dict[str, MoEModelConfig] = {
    "mixtral-8x7b": MoEModelConfig(
        num_experts=8,
        top_k=2,
        hidden_size=4096,
        intermediate_size=14336,
        model_name="mixtral-8x7b",
    ),
    "mixtral-8x22b": MoEModelConfig(
        num_experts=8,
        top_k=2,
        hidden_size=6144,
        intermediate_size=16384,
        model_name="mixtral-8x22b",
    ),
    "deepseek-v2-lite": MoEModelConfig(
        num_experts=64,
        top_k=6,
        hidden_size=2048,
        intermediate_size=1408,
        model_name="deepseek-v2-lite",
    ),
    "deepseek-v2": MoEModelConfig(
        num_experts=160,
        top_k=6,
        hidden_size=5120,
        intermediate_size=1536,
        model_name="deepseek-v2",
    ),
    "deepseek-v3": MoEModelConfig(
        num_experts=256,
        top_k=8,
        hidden_size=7168,
        intermediate_size=2048,
        model_name="deepseek-v3",
    ),
    "qwen2-57b-a14b": MoEModelConfig(
        num_experts=64,
        top_k=8,
        hidden_size=3584,
        intermediate_size=2560,
        model_name="qwen2-57b-a14b",
    ),
    "qwen3-235b": MoEModelConfig(
        num_experts=128,
        top_k=8,
        hidden_size=4096,
        intermediate_size=3072,
        model_name="qwen3-235b",
    ),
}


def _getattr_first(obj, names: list[str], default=None):
    """Return the first attribute found from a list of names."""
    for name in names:
        val = getattr(obj, name, None)
        if val is not None:
            return val
    return default


def load_from_hf(model_name_or_path: str) -> MoEModelConfig:
    """Load MoE config from a HuggingFace model (downloads config.json only)."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )

    num_experts = _getattr_first(
        config,
        ["num_local_experts", "n_routed_experts", "num_experts", "moe_num_experts"],
    )
    top_k = _getattr_first(
        config,
        ["num_experts_per_tok", "moe_top_k", "moe_topk"],
    )
    hidden_size = config.hidden_size
    intermediate_size = _getattr_first(
        config,
        ["moe_intermediate_size", "intermediate_size"],
    )

    if num_experts is None or top_k is None or intermediate_size is None:
        raise ValueError(
            f"Could not extract MoE parameters from {model_name_or_path}. "
            f"Got num_experts={num_experts}, top_k={top_k}, "
            f"intermediate_size={intermediate_size}. "
            "Is this an MoE model?"
        )

    return MoEModelConfig(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        model_name=model_name_or_path,
    )


def get_config(
    model: str | None = None,
    hf_model: str | None = None,
    num_experts: int | None = None,
    top_k: int | None = None,
    hidden_size: int | None = None,
    intermediate_size: int | None = None,
) -> MoEModelConfig:
    """Resolve a model config from preset name, HF model, or manual overrides."""
    if model and model in PRESETS:
        cfg = PRESETS[model]
    elif hf_model:
        cfg = load_from_hf(hf_model)
    elif all(v is not None for v in [num_experts, top_k, hidden_size, intermediate_size]):
        cfg = MoEModelConfig(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
    else:
        raise ValueError(
            "Provide --model <preset>, --hf-model <name>, or all of "
            "--num-experts/--top-k/--hidden-size/--intermediate-size"
        )

    # Allow per-field overrides on top of preset/HF config
    if num_experts is not None:
        cfg.num_experts = num_experts
    if top_k is not None:
        cfg.top_k = top_k
    if hidden_size is not None:
        cfg.hidden_size = hidden_size
    if intermediate_size is not None:
        cfg.intermediate_size = intermediate_size

    return cfg


def list_presets() -> None:
    """Print all available preset configs."""
    print(f"{'Name':<20} {'Experts':>8} {'Top-K':>6} {'Hidden':>7} {'Inter':>7}")
    print("-" * 52)
    for name, cfg in PRESETS.items():
        print(
            f"{name:<20} {cfg.num_experts:>8} {cfg.top_k:>6} "
            f"{cfg.hidden_size:>7} {cfg.intermediate_size:>7}"
        )
