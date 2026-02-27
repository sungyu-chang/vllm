# %% [markdown]
# # Expert Selection Distribution Analysis
#
# Analyses the expert routing logs produced by `02_run_inference.py`.
#
# ## Log format recap
# Each line in `expert_log.jsonl` is a JSON object:
# ```
# {
#   id, dataset_name, num_prompt_tokens, num_output_tokens,
#   total_tokens, num_layers, topk,
#   experts_shape: [total_tokens, num_layers, topk],
#   experts_b64: base64(int32 array)
# }
# ```
#
# ## Figures produced
# 1. **Aggregated CDF of expert choice across datasets** (one line per dataset)
# 2. **Aggregated CDF of expert choice at a specific layer** (one line per dataset, configurable layer ID)

# %%
import base64
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Configuration ────────────────────────────────────────────────────────────
LOG_FILE = "expert_log.jsonl"   # produced by 02_run_inference.py
LAYER_ID = 0                    # change to inspect a different layer (3-2)
# ─────────────────────────────────────────────────────────────────────────────

# %% [markdown]
# ## 1. Load and decode the log file

# %%
def decode_experts(record: dict) -> np.ndarray | None:
    """Decode the base64 expert array from a log record.
    Returns int32 array of shape [total_tokens, num_layers, topk], or None.
    """
    if record.get("experts_b64") is None:
        return None
    raw = base64.b64decode(record["experts_b64"])
    arr = np.frombuffer(raw, dtype=np.int32)
    return arr.reshape(record["experts_shape"])


def load_log(log_path: str) -> tuple[dict[str, list[np.ndarray]], dict[str, dict]]:
    """
    Read expert_log.jsonl and group expert arrays by dataset name.

    Returns:
        by_dataset : dict mapping dataset_name -> list of arrays
                     Each array has shape [total_tokens, num_layers, topk].
                     experts_b64 contains ONLY routed expert IDs.
        model_meta : dict with keys n_routed_experts, n_shared_experts
                     (inferred from the first valid record in the file;
                      shared experts are always-active and NOT present in
                      the arrays — they fire on every token for every layer).
    """
    by_dataset: dict[str, list[np.ndarray]] = defaultdict(list)
    model_meta: dict = {"n_routed_experts": None, "n_shared_experts": 0}
    skipped = 0

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            # Capture model-level metadata from the first record that has it
            if model_meta["n_routed_experts"] is None and rec.get("n_routed_experts"):
                model_meta["n_routed_experts"] = rec["n_routed_experts"]
                model_meta["n_shared_experts"] = rec.get("n_shared_experts", 0) or 0

            arr = decode_experts(rec)
            if arr is None:
                skipped += 1
                continue
            by_dataset[rec["dataset_name"]].append(arr)

    print(f"Loaded records per dataset:")
    for name, arrays in by_dataset.items():
        total_tok = sum(a.shape[0] for a in arrays)
        print(f"  {name:20s}: {len(arrays):5d} samples, {total_tok:8d} total tokens")
    if skipped:
        print(f"  (skipped {skipped} records with no expert data)")

    n_r = model_meta["n_routed_experts"]
    n_s = model_meta["n_shared_experts"]
    print(f"\nModel expert config: n_routed_experts={n_r}, n_shared_experts={n_s}")
    if n_s:
        print(f"  NOTE: {n_s} shared expert(s) are always active and are NOT "
              f"included in the logged arrays. Plots show routed experts only.")

    return dict(by_dataset), model_meta


by_dataset, model_meta = load_log(LOG_FILE)
N_SHARED = model_meta["n_shared_experts"]   # 0 for non-DeepSeek models

# %% [markdown]
# ## Helper: compute expert frequency distribution

# %%
def expert_counts_all_layers(arrays: list[np.ndarray]) -> np.ndarray:
    """
    Aggregate expert selection counts across ALL tokens and ALL layers.

    Args:
        arrays: list of [total_tokens, num_layers, topk] int32 arrays

    Returns:
        1-D int64 array indexed by expert ID with the total selection count.
        Contains ONLY routed expert IDs; shared experts are not represented.
    """
    if not arrays:
        return np.array([], dtype=np.int64)

    max_expert = max(a.max() for a in arrays)
    counts = np.zeros(max_expert + 1, dtype=np.int64)
    for a in arrays:
        flat = a.ravel()
        np.add.at(counts, flat, 1)
    return counts


def expert_counts_one_layer(arrays: list[np.ndarray], layer_id: int) -> np.ndarray:
    """
    Aggregate expert selection counts for a SINGLE layer.

    Args:
        arrays:   list of [total_tokens, num_layers, topk] int32 arrays
        layer_id: which layer to inspect (0-indexed)

    Returns:
        1-D int64 array indexed by expert ID.
        Contains ONLY routed expert IDs; shared experts are not represented.
    """
    if not arrays:
        return np.array([], dtype=np.int64)

    valid = [a for a in arrays if a.shape[1] > layer_id]
    if not valid:
        print(f"No arrays have layer {layer_id}")
        return np.array([], dtype=np.int64)

    max_expert = max(a[:, layer_id, :].max() for a in valid)
    counts = np.zeros(max_expert + 1, dtype=np.int64)
    for a in valid:
        flat = a[:, layer_id, :].ravel()   # shape [total_tokens * topk]
        np.add.at(counts, flat, 1)
    return counts


def counts_to_cdf(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a count array into a CDF over expert IDs (natural order).

    Returns:
        expert_ids: expert IDs 0, 1, …, N-1 (x-axis)
        cdf:        cumulative probability up to and including each expert (y-axis)
    """
    if len(counts) == 0:
        return np.array([]), np.array([])
    total = counts.sum()
    expert_ids = np.arange(len(counts))
    cdf = np.cumsum(counts) / total
    return expert_ids, cdf


def counts_to_pmf(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a count array into a PMF (probability mass per expert ID)."""
    if len(counts) == 0:
        return np.array([]), np.array([])
    total = counts.sum()
    expert_ids = np.arange(len(counts))
    pmf = counts / total
    return expert_ids, pmf


def add_shared_expert_footnote(fig, n_shared: int) -> None:
    """Add a small footnote to the bottom of the figure if n_shared > 0."""
    if n_shared <= 0:
        return
    note = (f"Note: {n_shared} shared expert(s) always active on every token "
            f"— not included in these counts (routed experts only).")
    fig.text(0.01, 0.01, note, fontsize=8, color="dimgray",
             va="bottom", ha="left", style="italic")
    fig.subplots_adjust(bottom=0.12)

# %% [markdown]
# ## 3-1. Aggregated CDF of expert choice across all layers (one line per dataset)

# %%
fig, ax = plt.subplots(figsize=(9, 5))

for dataset_name, arrays in by_dataset.items():
    counts = expert_counts_all_layers(arrays)
    expert_ids, cdf = counts_to_cdf(counts)
    if len(expert_ids) == 0:
        continue
    ax.plot(expert_ids, cdf, label=dataset_name, linewidth=1.5)

ax.set_xlabel("Expert ID (routed experts only)", fontsize=12)
ax.set_ylabel("Cumulative selection probability", fontsize=12)
ax.set_title("Aggregated CDF of Expert Selection (all layers)", fontsize=13)
ax.legend(loc="lower right", fontsize=10)
ax.set_xlim(left=0)
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.grid(True, linestyle="--", alpha=0.4)
add_shared_expert_footnote(fig, N_SHARED)
fig.tight_layout()
plt.savefig("fig_cdf_all_layers.png", dpi=150)
plt.savefig("fig_cdf_all_layers.pdf")
plt.show()

# %% [markdown]
# ## 3-2. CDF of expert choice at a specific layer (change `LAYER_ID` at the top)

# %%
fig, ax = plt.subplots(figsize=(9, 5))

for dataset_name, arrays in by_dataset.items():
    counts = expert_counts_one_layer(arrays, LAYER_ID)
    expert_ids, cdf = counts_to_cdf(counts)
    if len(expert_ids) == 0:
        continue
    ax.plot(expert_ids, cdf, label=dataset_name, linewidth=1.5)

ax.set_xlabel("Expert ID (routed experts only)", fontsize=12)
ax.set_ylabel("Cumulative selection probability", fontsize=12)
ax.set_title(f"Aggregated CDF of Expert Selection at Layer {LAYER_ID}", fontsize=13)
ax.legend(loc="lower right", fontsize=10)
ax.set_xlim(left=0)
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.grid(True, linestyle="--", alpha=0.4)
add_shared_expert_footnote(fig, N_SHARED)
fig.tight_layout()
plt.savefig(f"fig_cdf_layer{LAYER_ID}.png", dpi=150)
plt.savefig(f"fig_cdf_layer{LAYER_ID}.pdf")
plt.show()

# %% [markdown]
# ## 4-1. Probability distribution (PMF) of expert selection — all layers
#
# Each expert's share of total selections, shown as a probability mass function.

# %%
fig, ax = plt.subplots(figsize=(9, 5))

for dataset_name, arrays in by_dataset.items():
    counts = expert_counts_all_layers(arrays)
    expert_ids, pmf = counts_to_pmf(counts)
    if len(expert_ids) == 0:
        continue
    ax.plot(expert_ids, pmf, label=dataset_name, linewidth=1.2, alpha=0.85)

if by_dataset:
    n_experts = max(len(expert_counts_all_layers(arrs)) for arrs in by_dataset.values())
    ax.axhline(1 / n_experts, color="black", linestyle=":", linewidth=1, label="Uniform (routed)")

ax.set_xlabel("Expert ID (routed experts only)", fontsize=12)
ax.set_ylabel("Selection probability", fontsize=12)
ax.set_title("PMF of Expert Selection (all layers)", fontsize=13)
ax.legend(loc="upper right", fontsize=10)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.grid(True, linestyle="--", alpha=0.4)
add_shared_expert_footnote(fig, N_SHARED)
fig.tight_layout()
plt.savefig("fig_pmf_all_layers.png", dpi=150)
plt.savefig("fig_pmf_all_layers.pdf")
plt.show()

# %% [markdown]
# ## 4-2. PMF of expert selection at a specific layer (change `LAYER_ID` at the top)

# %%
fig, ax = plt.subplots(figsize=(9, 5))

for dataset_name, arrays in by_dataset.items():
    counts = expert_counts_one_layer(arrays, LAYER_ID)
    expert_ids, pmf = counts_to_pmf(counts)
    if len(expert_ids) == 0:
        continue
    ax.plot(expert_ids, pmf, label=dataset_name, linewidth=1.2, alpha=0.85)

if by_dataset:
    n_experts = max(
        len(expert_counts_one_layer(arrs, LAYER_ID)) for arrs in by_dataset.values()
    )
    if n_experts > 0:
        ax.axhline(1 / n_experts, color="black", linestyle=":", linewidth=1, label="Uniform (routed)")

ax.set_xlabel("Expert ID (routed experts only)", fontsize=12)
ax.set_ylabel("Selection probability", fontsize=12)
ax.set_title(f"PMF of Expert Selection at Layer {LAYER_ID}", fontsize=13)
ax.legend(loc="upper right", fontsize=10)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.grid(True, linestyle="--", alpha=0.4)
add_shared_expert_footnote(fig, N_SHARED)
fig.tight_layout()
plt.savefig(f"fig_pmf_layer{LAYER_ID}.png", dpi=150)
plt.savefig(f"fig_pmf_layer{LAYER_ID}.pdf")
plt.show()

# %% [markdown]
# ## 4-3. PMF (line) and absolute invocation count (scatter) — all layers

# %%
# ── PMF: line figure ──────────────────────────────────────────────────────────
fig_pmf, ax_pmf = plt.subplots(figsize=(9, 5))

for dataset_name, arrays in by_dataset.items():
    counts = expert_counts_all_layers(arrays)
    expert_ids, pmf = counts_to_pmf(counts)
    if len(expert_ids) == 0:
        continue
    ax_pmf.plot(expert_ids, pmf, label=dataset_name, linewidth=1.2, alpha=0.85)

if by_dataset:
    n_experts = max(len(expert_counts_all_layers(arrs)) for arrs in by_dataset.values())
    if n_experts > 0:
        ax_pmf.axhline(1 / n_experts, color="black", linestyle=":", linewidth=1, label="Uniform")

ax_pmf.set_xlabel("Expert ID", fontsize=12)
ax_pmf.set_ylabel("Selection probability", fontsize=12)
ax_pmf.set_title("PMF of Expert Selection (all layers)", fontsize=13)
ax_pmf.legend(loc="upper right", fontsize=10)
ax_pmf.set_xlim(left=0)
ax_pmf.set_ylim(bottom=0)
ax_pmf.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax_pmf.grid(True, linestyle="--", alpha=0.4)
fig_pmf.tight_layout()
fig_pmf.savefig("fig_pmf_all_layers_v2.png", dpi=150)
fig_pmf.savefig("fig_pmf_all_layers_v2.pdf")
plt.show()

# ── Absolute count: scatter dot figure ───────────────────────────────────────
fig_cnt, ax_cnt = plt.subplots(figsize=(9, 5))

for dataset_name, arrays in by_dataset.items():
    counts = expert_counts_all_layers(arrays)
    expert_ids = np.arange(len(counts))
    if len(expert_ids) == 0:
        continue
    ax_cnt.scatter(expert_ids, counts, label=dataset_name, s=8, alpha=0.7)

if by_dataset:
    n_experts = max(len(expert_counts_all_layers(arrs)) for arrs in by_dataset.values())
    total_all = sum(expert_counts_all_layers(arrs).sum() for arrs in by_dataset.values())
    if n_experts > 0:
        ax_cnt.axhline(total_all / n_experts, color="black", linestyle=":", linewidth=1, label="Uniform")

ax_cnt.set_xlabel("Expert ID", fontsize=12)
ax_cnt.set_ylabel("Number of invocations", fontsize=12)
ax_cnt.set_title("Absolute Invocation Count per Expert (all layers)", fontsize=13)
ax_cnt.legend(loc="upper right", fontsize=10)
ax_cnt.set_xlim(left=0)
ax_cnt.set_ylim(bottom=0)
ax_cnt.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax_cnt.grid(True, linestyle="--", alpha=0.4)
fig_cnt.tight_layout()
fig_cnt.savefig("fig_count_all_layers.png", dpi=150)
fig_cnt.savefig("fig_count_all_layers.pdf")
plt.show()

# %% [markdown]
# ## 4-4. Per-layer PMF (line) and absolute count (scatter) — saved to `layer_figures/`
#
# One PMF figure and one count figure per layer, both saved as PNG and PDF.

# %%
all_arrays = [a for arrs in by_dataset.values() for a in arrs]

if not all_arrays:
    print("No data loaded – run 02_run_inference.py first.")
else:
    num_layers = max(a.shape[1] for a in all_arrays)
    print(f"Model has {num_layers} MoE layers. Generating figures …")

    out_dir = Path("layer_figures")
    out_dir.mkdir(exist_ok=True)

    for layer in range(num_layers):
        # ── PMF: line figure ──────────────────────────────────────────────
        fig_pmf, ax_pmf = plt.subplots(figsize=(9, 5))
        n_experts_layer = 0

        for dataset_name, arrays in by_dataset.items():
            counts = expert_counts_one_layer(arrays, layer)
            expert_ids, pmf = counts_to_pmf(counts)
            if len(expert_ids) == 0:
                continue
            n_experts_layer = max(n_experts_layer, len(counts))
            ax_pmf.plot(expert_ids, pmf, label=dataset_name, linewidth=1.2, alpha=0.85)

        if n_experts_layer > 0:
            ax_pmf.axhline(1 / n_experts_layer, color="black", linestyle=":",
                           linewidth=1, label="Uniform")

        ax_pmf.set_xlabel("Expert ID", fontsize=12)
        ax_pmf.set_ylabel("Selection probability", fontsize=12)
        ax_pmf.set_title(f"PMF of Expert Selection — Layer {layer}", fontsize=13)
        ax_pmf.legend(loc="upper right", fontsize=10)
        ax_pmf.set_xlim(left=0)
        ax_pmf.set_ylim(bottom=0)
        ax_pmf.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax_pmf.grid(True, linestyle="--", alpha=0.4)
        fig_pmf.tight_layout()
        fig_pmf.savefig(out_dir / f"pmf_layer{layer:02d}.png", dpi=150)
        fig_pmf.savefig(out_dir / f"pmf_layer{layer:02d}.pdf")
        plt.close(fig_pmf)

        # ── Absolute count: scatter figure ────────────────────────────────
        fig_cnt, ax_cnt = plt.subplots(figsize=(9, 5))
        total_layer = 0

        for dataset_name, arrays in by_dataset.items():
            counts = expert_counts_one_layer(arrays, layer)
            expert_ids = np.arange(len(counts))
            if len(expert_ids) == 0:
                continue
            total_layer += counts.sum()
            ax_cnt.scatter(expert_ids, counts, label=dataset_name, s=8, alpha=0.7)

        if n_experts_layer > 0 and total_layer > 0:
            ax_cnt.axhline(total_layer / n_experts_layer, color="black", linestyle=":",
                           linewidth=1, label="Uniform")

        ax_cnt.set_xlabel("Expert ID", fontsize=12)
        ax_cnt.set_ylabel("Number of invocations", fontsize=12)
        ax_cnt.set_title(f"Absolute Invocation Count per Expert — Layer {layer}", fontsize=13)
        ax_cnt.legend(loc="upper right", fontsize=10)
        ax_cnt.set_xlim(left=0)
        ax_cnt.set_ylim(bottom=0)
        ax_cnt.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax_cnt.grid(True, linestyle="--", alpha=0.4)
        fig_cnt.tight_layout()
        fig_cnt.savefig(out_dir / f"count_layer{layer:02d}.png", dpi=150)
        fig_cnt.savefig(out_dir / f"count_layer{layer:02d}.pdf")
        plt.close(fig_cnt)

    print(f"Saved {num_layers * 2} figures (PMF + count per layer) to '{out_dir}/'")
    print(f"  pmf_layer00.png … pmf_layer{num_layers-1:02d}.png")
    print(f"  count_layer00.png … count_layer{num_layers-1:02d}.png")

# %% [markdown]
# ## Bonus: interactive layer sweep
# Run the cell below to save one CDF figure per layer.

# %%
all_arrays = [a for arrs in by_dataset.values() for a in arrs]
if all_arrays:
    num_layers = max(a.shape[1] for a in all_arrays)
    print(f"Model has {num_layers} MoE layers captured.")

    out_dir = Path("layer_cdfs")
    out_dir.mkdir(exist_ok=True)

    for layer in range(num_layers):
        fig, ax = plt.subplots(figsize=(9, 5))
        for dataset_name, arrays in by_dataset.items():
            counts = expert_counts_one_layer(arrays, layer)
            expert_ids, cdf = counts_to_cdf(counts)
            if len(expert_ids) == 0:
                continue
            ax.plot(expert_ids, cdf, label=dataset_name, linewidth=1.5)

        ax.set_xlabel("Expert ID", fontsize=12)
        ax.set_ylabel("Cumulative selection probability", fontsize=12)
        ax.set_title(f"CDF of Expert Selection — Layer {layer}", fontsize=13)
        ax.legend(loc="lower right", fontsize=10)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out_dir / f"cdf_layer{layer:02d}.png", dpi=150)
        fig.savefig(out_dir / f"cdf_layer{layer:02d}.pdf")
        plt.close(fig)

    print(f"Saved {num_layers} figures to {out_dir}/")
else:
    print("No data loaded – run 02_run_inference.py first.")

# %% [markdown]
# ## Combined PDF report

# %%
from matplotlib.backends.backend_pdf import PdfPages

COMBINED_PDF = "expert_selection_report.pdf"


def _make_cdf_ax(ax, dataset_dict, counts_fn, title):
    for dataset_name, arrays in dataset_dict.items():
        counts = counts_fn(arrays)
        expert_ids, cdf = counts_to_cdf(counts)
        if len(expert_ids) == 0:
            continue
        ax.plot(expert_ids, cdf, label=dataset_name, linewidth=1.5)
    ax.set_xlabel("Expert ID", fontsize=12)
    ax.set_ylabel("Cumulative selection probability", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(left=0); ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(True, linestyle="--", alpha=0.4)


def _make_pmf_ax(ax, dataset_dict, counts_fn, title):
    all_counts = [counts_fn(arrs) for arrs in dataset_dict.values()]
    n_experts = max((len(c) for c in all_counts if len(c) > 0), default=0)
    for dataset_name, arrays in dataset_dict.items():
        counts = counts_fn(arrays)
        expert_ids, pmf = counts_to_pmf(counts)
        if len(expert_ids) == 0:
            continue
        ax.plot(expert_ids, pmf, label=dataset_name, linewidth=1.2, alpha=0.85)
    if n_experts > 0:
        ax.axhline(1 / n_experts, color="black", linestyle=":", linewidth=1, label="Uniform")
    ax.set_xlabel("Expert ID", fontsize=12)
    ax.set_ylabel("Selection probability", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(True, linestyle="--", alpha=0.4)


with PdfPages(COMBINED_PDF) as pdf:
    # CDF — all layers
    fig, ax = plt.subplots(figsize=(9, 5))
    _make_cdf_ax(ax, by_dataset, expert_counts_all_layers,
                 "Aggregated CDF of Expert Selection (all layers)")
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # PMF — all layers
    fig, ax = plt.subplots(figsize=(9, 5))
    _make_pmf_ax(ax, by_dataset, expert_counts_all_layers,
                 "PMF of Expert Selection (all layers)")
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Per-layer CDF + PMF
    all_arrays = [a for arrs in by_dataset.values() for a in arrs]
    if all_arrays:
        num_layers = max(a.shape[1] for a in all_arrays)
        for layer in range(num_layers):
            layer_fn = lambda arrs, l=layer: expert_counts_one_layer(arrs, l)

            fig, ax = plt.subplots(figsize=(9, 5))
            _make_cdf_ax(ax, by_dataset, layer_fn,
                         f"CDF of Expert Selection — Layer {layer}")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

            fig, ax = plt.subplots(figsize=(9, 5))
            _make_pmf_ax(ax, by_dataset, layer_fn,
                         f"PMF of Expert Selection — Layer {layer}")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    meta = pdf.infodict()
    meta["Title"] = "Expert Selection Distribution Report"
    meta["Subject"] = "MoE expert routing CDF and PMF by dataset and layer"

print(f"Combined PDF saved to: {COMBINED_PDF}")
