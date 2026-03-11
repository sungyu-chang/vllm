# SPDX-License-Identifier: Apache-2.0
"""Generate expert routing assignments with controllable skewness."""

import numpy as np
import torch


def generate_expert_assignments(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    distribution: str = "uniform",
    zipf_alpha: float = 1.0,
    hot_expert_id: int = 0,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate (topk_ids, topk_weights) with controllable expert skewness.

    Args:
        num_tokens: Number of input tokens (M).
        num_experts: Total number of experts (E).
        top_k: Number of experts selected per token.
        distribution: One of "uniform", "zipf", "single_hot".
        zipf_alpha: Exponent for Zipf distribution (0=uniform, higher=more skewed).
        hot_expert_id: Which expert gets all tokens in "single_hot" mode.
        seed: Random seed for reproducibility.

    Returns:
        topk_ids:     [num_tokens, top_k] int32 tensor on CUDA
        topk_weights: [num_tokens, top_k] float32 tensor on CUDA (normalized per row)
    """
    rng = np.random.default_rng(seed)

    if distribution == "uniform":
        probs = np.ones(num_experts, dtype=np.float64)
    elif distribution == "zipf":
        ranks = np.arange(1, num_experts + 1, dtype=np.float64)
        probs = 1.0 / np.power(ranks, zipf_alpha)
    elif distribution == "single_hot":
        probs = np.zeros(num_experts, dtype=np.float64)
        probs[hot_expert_id] = 1.0
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    probs = probs / probs.sum()

    # Sample top_k experts per token (without replacement within each token)
    topk_ids = np.empty((num_tokens, top_k), dtype=np.int32)

    if distribution == "single_hot":
        topk_ids[:, 0] = hot_expert_id
        if top_k > 1:
            others = [e for e in range(num_experts) if e != hot_expert_id]
            for i in range(num_tokens):
                topk_ids[i, 1:] = rng.choice(others, size=top_k - 1, replace=False)
    else:
        for i in range(num_tokens):
            topk_ids[i] = rng.choice(num_experts, size=top_k, replace=False, p=probs)

    # Generate random weights and normalize per row
    raw_weights = rng.random((num_tokens, top_k)).astype(np.float32)
    row_sums = raw_weights.sum(axis=1, keepdims=True)
    topk_weights = raw_weights / row_sums

    topk_ids_t = torch.from_numpy(topk_ids).to(dtype=torch.int32, device="cuda")
    topk_weights_t = torch.from_numpy(topk_weights).to(dtype=torch.float32, device="cuda")

    return topk_ids_t, topk_weights_t


def get_distribution_stats(
    topk_ids: torch.Tensor, num_experts: int
) -> dict:
    """Compute expert load statistics."""
    flat = topk_ids.view(-1).cpu()
    counts = torch.zeros(num_experts, dtype=torch.int64)
    for eid in range(num_experts):
        counts[eid] = (flat == eid).sum().item()

    total = counts.sum().item()
    counts_np = counts.numpy()

    # Gini coefficient
    sorted_counts = np.sort(counts_np)
    n = len(sorted_counts)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_counts) / (n * np.sum(sorted_counts)) - (n + 1) / n) if np.sum(sorted_counts) > 0 else 0.0

    return {
        "counts": counts_np,
        "total": total,
        "min": int(counts_np.min()),
        "max": int(counts_np.max()),
        "mean": float(counts_np.mean()),
        "std": float(counts_np.std()),
        "gini": float(gini),
        "num_active": int((counts_np > 0).sum()),
    }


def print_distribution_stats(topk_ids: torch.Tensor, num_experts: int) -> None:
    """Print expert load distribution summary."""
    stats = get_distribution_stats(topk_ids, num_experts)
    print(f"  Total token-expert pairs: {stats['total']}")
    print(f"  Active experts: {stats['num_active']}/{num_experts}")
    print(
        f"  Tokens/expert: min={stats['min']}, max={stats['max']}, "
        f"mean={stats['mean']:.1f}, std={stats['std']:.1f}"
    )
    print(f"  Gini coefficient: {stats['gini']:.3f}")

    # Show top-10 and bottom-5 expert loads
    counts = stats["counts"]
    sorted_idx = np.argsort(counts)[::-1]
    n_show = min(10, num_experts)
    top = [(int(sorted_idx[i]), int(counts[sorted_idx[i]])) for i in range(n_show)]
    print(f"  Top-{n_show} experts: {top}")
