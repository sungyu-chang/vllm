# %% [markdown]
# # Expert Selection Distribution Analysis
#
# Analyses the expert routing logs produced by `02_run_inference.py`.
#
# ## Log format recap
# Each line in `expert_log_{model_name}.jsonl` is a JSON object:
# ```
# {
#   id, dataset_name, num_prompt_tokens, num_output_tokens,
#   total_tokens, num_layers, topk,
#   experts_shape: [total_tokens, num_layers, topk],
#   experts_b64: base64(int32 array)
# }
# ```
#
# ## Analyses produced
# 1. **Aggregated CDF / PMF of expert choice across datasets**
# 2. **Per-layer CDF / PMF figures**
# 3. **KL divergence** — pairwise comparison of PMFs (overall + per layer)
# 4. **Expert continuity** — PMF of consecutive-layer run lengths per dataset

# %%
import base64
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from cycler import cycler
import pathlib, textwrap

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_NAME = "deepseek"      # ← set your model name here
# Override from command line:  python 03_analysis.py <model_name>
if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    MODEL_NAME = sys.argv[1]

LOG_FILE = f"expert_log_{MODEL_NAME}.jsonl"
LAYER_ID = 0                      # change to inspect a different layer (3-2)

OUT_DIR = Path(f"figures_{MODEL_NAME}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


"""Academic matplotlib style setup.

Import this module or paste its contents into a script / notebook to get:

- Journal-ready rcParams (Arial Bold, RColorBrewer "Paired" palette, …)
- ``fp`` — a ``FontProperties`` object (Arial Bold 16) for explicit use
- ``PAIRED``, ``MARKERS``, ``LINESTYLES`` — convenience constants
- ``style_ax(ax)`` / ``style_fig(fig, …)`` — apply ``fp`` to all text and
  optionally create a top or inside legend in one call

Requires: matplotlib, cycler.  For Arial on Ubuntu run ``install-fonts.sh``.
"""

import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import font_manager

# ── Academic style (applied via rcParams) ───────────────────────────────────
_ACADEMIC_RC = {
    "pdf.fonttype":      42,
    "ps.fonttype":       42,

    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "DejaVu Sans", "Helvetica", "Liberation Sans"],
    "font.weight":       "bold",
    "font.size":         16,
    "axes.titlesize":    16,
    "axes.labelsize":    16,
    "xtick.labelsize":   16,
    "ytick.labelsize":   16,
    "legend.fontsize":   14,

    "axes.prop_cycle":   cycler("color", [
        "#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C", "#FB9A99", "#E31A1C",
        "#FDBF6F", "#FF7F00", "#CAB2D6", "#6A3D9A", "#FFFF99", "#B15928",
    ]),

    "lines.linewidth":   1.5,
    "lines.markersize":  12,

    "axes.linewidth":    1.5,
    "axes.edgecolor":    "black",
    "axes.axisbelow":    True,

    "xtick.major.width": 1.5,
    "xtick.major.size":  3,
    "ytick.major.width": 1.5,
    "ytick.major.size":  3,
    "xtick.minor.width": 1.0,
    "xtick.minor.size":  2,
    "ytick.minor.width": 1.0,
    "ytick.minor.size":  2,
    "xtick.direction":   "out",
    "ytick.direction":   "out",

    "axes.grid":         True,
    "grid.linestyle":    "--",
    "grid.linewidth":    1.0,
    "grid.alpha":        0.6,
    "axes.grid.axis":    "y",

    "legend.frameon":       False,
    "legend.handlelength":  1.5,
    "legend.handletextpad": 0.4,
    "legend.columnspacing": 1.0,

    "hatch.linewidth":  0.5,

    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.01,
}
matplotlib.rcParams.update(_ACADEMIC_RC)

# ── Font (Arial Bold 16) ───────────────────────────────────────────────────
fp_path = font_manager.findfont(font_manager.FontProperties(family="Arial"))
fp      = font_manager.FontProperties(fname=fp_path, weight="bold", size=16)

# ── RColorBrewer "Paired" ── C0-C11 ─────────────────────────────────────────
# Code  Name       Hex
# C0    BLUE_LT    #A6CEE3
# C1    BLUE_DK    #1F78B4
# C2    GREEN_LT   #B2DF8A
# C3    GREEN_DK   #33A02C
# C4    RED_LT     #FB9A99
# C5    RED_DK     #E31A1C
# C6    ORANGE_LT  #FDBF6F
# C7    ORANGE_DK  #FF7F00
# C8    PURPLE_LT  #CAB2D6
# C9    PURPLE_DK  #6A3D9A
# C10   YELLOW_LT  #FFFF99
# C11   BROWN_DK   #B15928
BLUE_LT   = "#A6CEE3"; BLUE_DK   = "#1F78B4"
GREEN_LT  = "#B2DF8A"; GREEN_DK  = "#33A02C"
RED_LT    = "#FB9A99"; RED_DK    = "#E31A1C"
ORANGE_LT = "#FDBF6F"; ORANGE_DK = "#FF7F00"
PURPLE_LT = "#CAB2D6"; PURPLE_DK = "#6A3D9A"
YELLOW_LT = "#FFFF99"; BROWN_DK  = "#B15928"
PAIRED    = [BLUE_LT, BLUE_DK, GREEN_LT, GREEN_DK, RED_LT, RED_DK,
             ORANGE_LT, ORANGE_DK, PURPLE_LT, PURPLE_DK, YELLOW_LT, BROWN_DK]

MARKERS    = ["s", "D", "^", "d", "o", "v", "P", "X"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]


def style_ax(ax, _fp=None, enforce=False):
    """Apply fp (Arial Bold 16) to all text elements on a single Axes.

    Parameters
    ----------
    enforce : bool, default False
        When True, also re-apply all academic rcParams (ticks, grid, spines,
        line widths, marker sizes) directly onto the axes — useful for
        overriding styles set by someone else's script.

    Call after all data, labels, and limits are set, just before
    tight_layout() / savefig().
    """
    _fp = _fp or fp
    rc = _ACADEMIC_RC

    # ── Fonts (always applied) ──
    ax.xaxis.label.set_fontproperties(_fp)
    ax.yaxis.label.set_fontproperties(_fp)
    ax.title.set_fontproperties(_fp)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(_fp)
    leg = ax.get_legend()
    if leg is not None:
        for t in leg.get_texts():
            t.set_fontproperties(_fp)

    if not enforce:
        return

    # ── Spines ──
    for spine in ax.spines.values():
        spine.set_linewidth(rc["axes.linewidth"])
        spine.set_edgecolor(rc["axes.edgecolor"])

    # ── Ticks ──
    ax.tick_params(axis="both", which="major",
                   direction=rc["xtick.direction"],
                   width=rc["xtick.major.width"],
                   length=rc["xtick.major.size"],
                   labelsize=rc["xtick.labelsize"])
    ax.tick_params(axis="both", which="minor",
                   direction=rc["xtick.direction"],
                   width=rc["xtick.minor.width"],
                   length=rc["xtick.minor.size"])

    # ── Grid ──
    ax.set_axisbelow(rc["axes.axisbelow"])
    grid_axis = rc["axes.grid.axis"]
    grid_kw = dict(linestyle=rc["grid.linestyle"],
                   linewidth=rc["grid.linewidth"],
                   alpha=rc["grid.alpha"])
    ax.xaxis.grid(grid_axis in ("x", "both"), **grid_kw)
    ax.yaxis.grid(grid_axis in ("y", "both"), **grid_kw)

    # ── Lines & markers on existing artists ──
    for line in ax.get_lines():
        line.set_linewidth(rc["lines.linewidth"])
        line.set_markersize(rc["lines.markersize"])


def style_fig(fig, _fp=None, legend_ncol=None, legend_level="fig",
              legend_loc="top", enforce=False, **legend_kw):
    """Apply fp to every Axes in fig and optionally create a legend.

    Parameters
    ----------
    legend_ncol : int, optional
        Number of legend columns.  When set, existing legends are replaced.
        Extra keyword arguments are forwarded to the legend call.
    legend_level : {"fig", "ax"}, default "fig"
        ``"fig"`` — one shared legend (handles collected from all axes,
        first occurrence per label wins).
        ``"ax"``  — one legend per axes that has labelled handles.
    legend_loc : str, default "top"
        ``"top"`` — place the legend outside, centred above the plot area.
        Any other matplotlib *loc* string (e.g. ``"best"``, ``"upper right"``)
        places the legend **inside** the axes / figure.
    enforce : bool, default False
        When True, also re-apply tick, grid, spine, and line-width settings
        onto every axes — useful for overriding someone else's script.

    Examples::

        style_fig(fig)                                          # font only
        style_fig(fig, legend_ncol=4)                           # shared, top
        style_fig(fig, legend_ncol=4, legend_loc="best")        # shared, auto inside
        style_fig(fig, legend_ncol=2, legend_level="ax")        # per-axes, top
        style_fig(fig, legend_ncol=2, legend_level="ax",
                  legend_loc="upper right")                     # per-axes, inside
        style_fig(fig, enforce=True)                            # override foreign style
    """
    _fp = _fp or fp
    if enforce:
        matplotlib.rcParams.update(_ACADEMIC_RC)
    for ax in fig.get_axes():
        style_ax(ax, _fp, enforce=enforce)

    if legend_ncol is None:
        for leg in fig.legends:
            for t in leg.get_texts():
                t.set_fontproperties(_fp)
        return

    if legend_level == "fig":
        # Collect handles/labels from all axes, first occurrence per label wins
        seen: dict = {}
        for ax in fig.get_axes():
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in seen:
                    seen[l] = h
        # Remove per-axes legends
        for ax in fig.get_axes():
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        if seen:
            if legend_loc == "top":
                kw = dict(ncol=legend_ncol, frameon=False, prop=_fp,
                          loc="upper center", bbox_to_anchor=(0.5, 1.0))
            else:
                kw = dict(ncol=legend_ncol, frameon=False, prop=_fp,
                          loc=legend_loc)
            kw.update(legend_kw)
            fig.legend(list(seen.values()), list(seen.keys()), **kw)

    elif legend_level == "ax":
        for ax in fig.get_axes():
            handles, labels = ax.get_legend_handles_labels()
            if not handles:
                continue
            if ax.get_legend() is not None:
                ax.get_legend().remove()
            if legend_loc == "top":
                kw = dict(ncol=legend_ncol, frameon=False, prop=_fp,
                          loc="lower center", bbox_to_anchor=(0.5, 1.0))
            else:
                kw = dict(ncol=legend_ncol, frameon=False, prop=_fp,
                          loc=legend_loc)
            kw.update(legend_kw)
            ax.legend(handles, labels, **kw)

print(f"Model:      {MODEL_NAME}")
print(f"Log file:   {LOG_FILE}")
print(f"Output dir: {OUT_DIR}/")
print()

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
# ## 2. KL Divergence Analysis
#
# Compare expert selection PMFs between datasets using symmetric
# KL divergence: SKL(P, Q) = (KL(P‖Q) + KL(Q‖P)) / 2.
#
# Reported for: (a) overall distribution across all layers, and
# (b) per-layer distributions with a heatmap summary.

# %%
def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-12) -> float:
    """KL(P ‖ Q) with additive smoothing to avoid log(0)."""
    p = np.asarray(p, dtype=np.float64) + epsilon
    q = np.asarray(q, dtype=np.float64) + epsilon
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def symmetric_kl(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-12) -> float:
    """Symmetric KL divergence: (KL(P‖Q) + KL(Q‖P)) / 2."""
    return (kl_divergence(p, q, epsilon) + kl_divergence(q, p, epsilon)) / 2.0


def pad_to_same_length(*arrays: np.ndarray) -> list[np.ndarray]:
    """Zero-pad arrays so they all share the same length."""
    max_len = max(len(a) for a in arrays)
    return [np.pad(a, (0, max_len - len(a))) if len(a) < max_len else a
            for a in arrays]


# ── 2a. Overall pairwise KL ─────────────────────────────────────────────────
dataset_names = list(by_dataset.keys())

if len(dataset_names) >= 2:
    dataset_pmfs_overall: dict[str, np.ndarray] = {}
    for name, arrays in by_dataset.items():
        counts = expert_counts_all_layers(arrays)
        _, pmf = counts_to_pmf(counts)
        dataset_pmfs_overall[name] = pmf

    pairs = list(combinations(dataset_names, 2))

    print("=" * 65)
    print("Pairwise Symmetric KL Divergence  (all layers aggregated)")
    print("=" * 65)
    for a, b in pairs:
        pa, pb = pad_to_same_length(dataset_pmfs_overall[a],
                                    dataset_pmfs_overall[b])
        skl = symmetric_kl(pa, pb)
        if skl >= 0.1:
            flag = "  *** significant"
        elif skl >= 0.01:
            flag = "  *   moderate"
        else:
            flag = ""
        print(f"  {a:20s} vs {b:20s}:  SKL = {skl:.6f}{flag}")
    print()
else:
    pairs = []
    print("(Only one dataset — skipping pairwise KL comparison.)\n")

# ── 2b. Per-layer pairwise KL + heatmap ─────────────────────────────────────
all_arrays = [a for arrs in by_dataset.values() for a in arrs]

if len(pairs) > 0 and all_arrays:
    num_layers = max(a.shape[1] for a in all_arrays)
    kl_matrix = np.zeros((len(pairs), num_layers))

    for i, (a, b) in enumerate(pairs):
        for layer in range(num_layers):
            ca = expert_counts_one_layer(by_dataset[a], layer)
            cb = expert_counts_one_layer(by_dataset[b], layer)
            _, pa = counts_to_pmf(ca)
            _, pb = counts_to_pmf(cb)
            if len(pa) == 0 or len(pb) == 0:
                continue
            pa, pb = pad_to_same_length(pa, pb)
            kl_matrix[i, layer] = symmetric_kl(pa, pb)

    print("Per-layer Symmetric KL Divergence")
    print("-" * 65)
    for i, (a, b) in enumerate(pairs):
        row = kl_matrix[i]
        sig_layers = np.where(row >= 0.1)[0]
        mod_layers = np.where((row >= 0.01) & (row < 0.1))[0]
        print(f"  {a} vs {b}:")
        print(f"    Mean SKL = {row.mean():.6f},  "
              f"Max SKL = {row.max():.6f} (layer {row.argmax()})")
        if len(sig_layers) > 0:
            print(f"    Significant layers (SKL >= 0.1): {sig_layers.tolist()}")
        if len(mod_layers) > 0:
            print(f"    Moderate layers  (0.01 <= SKL < 0.1): {mod_layers.tolist()}")
    print()

    # Heatmap
    fig, ax = plt.subplots(
        figsize=(max(10, num_layers * 0.3), max(3, len(pairs) * 0.8 + 1.5)))
    im = ax.imshow(kl_matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Dataset pair", fontsize=12)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([f"{a} vs {b}" for a, b in pairs], fontsize=9)
    ax.set_title(f"Per-layer Symmetric KL Divergence — {MODEL_NAME}", fontsize=13)
    fig.colorbar(im, ax=ax, label="Symmetric KL")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_kl_per_layer_heatmap.png", dpi=150)
    plt.show()

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
plt.savefig(OUT_DIR / "fig_cdf_all_layers.png", dpi=150)
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
plt.savefig(OUT_DIR / "fig_pmf_all_layers.png", dpi=150)
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
fig_pmf.savefig(OUT_DIR / "fig_pmf_all_layers_v2.png", dpi=150)
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
fig_cnt.savefig(OUT_DIR / "fig_count_all_layers.png", dpi=150)
plt.show()

# %% [markdown]
# ## 4-4. Per-layer PMF (line) and absolute count (scatter) — saved to `layer_figures/`
#
# One PMF figure and one count figure per layer, saved as PNG.

# %%
all_arrays = [a for arrs in by_dataset.values() for a in arrs]

if not all_arrays:
    print("No data loaded – run 02_run_inference.py first.")
else:
    num_layers = max(a.shape[1] for a in all_arrays)
    print(f"Model has {num_layers} MoE layers. Generating figures …")

    layer_fig_dir = OUT_DIR / "layer_figures"
    layer_fig_dir.mkdir(exist_ok=True)

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
        fig_pmf.savefig(layer_fig_dir / f"pmf_layer{layer:02d}.png", dpi=150)
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
        fig_cnt.savefig(layer_fig_dir / f"count_layer{layer:02d}.png", dpi=150)
        plt.close(fig_cnt)

    print(f"Saved {num_layers * 2} figures (PMF + count per layer) to '{layer_fig_dir}/'")
    print(f"  pmf_layer00.png … pmf_layer{num_layers-1:02d}.png")
    print(f"  count_layer00.png … count_layer{num_layers-1:02d}.png")


# %% [markdown]
# ## 5. Expert Continuity Analysis
#
# For every token in every sample, track which experts are selected at
# consecutive MoE layers.  A **run of length r** means an expert was
# continuously selected for **r consecutive layers** for a single token.
#
# We compute the PMF of run lengths per dataset and for all datasets
# combined, revealing whether experts tend to "stick" across layers or
# are reshuffled at each layer.

# %%
def compute_continuity_runs(arrays: list[np.ndarray],
                            n_experts: int) -> np.ndarray:
    """
    Count consecutive-layer run lengths for every expert at every token.

    For each token, an expert's "run" starts when it first appears in a
    layer's top-k selection and ends when it is absent.  The length of
    that run (number of consecutive layers) is tallied.

    Args:
        arrays:    list of [total_tokens, num_layers, topk] int32 arrays
        n_experts: number of routed experts (used for the presence matrix)

    Returns:
        1-D int64 array where counts[r] = number of runs of length r.
        counts[0] is always 0.
    """
    all_runs: list[np.ndarray] = []

    for arr in arrays:
        T, L, K = arr.shape
        # Process tokens in chunks to limit memory
        # Each chunk builds a [ct, L, n_experts] bool presence matrix
        chunk_size = max(1, min(500, 50_000_000 // (L * n_experts)))

        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            chunk = arr[start:end]          # [ct, L, K]
            ct = chunk.shape[0]

            # Build presence matrix: presence[t, l, e] = 1 iff expert e
            # is in the top-k at token t, layer l.
            presence = np.zeros((ct, L, n_experts), dtype=np.int8)
            t_idx = np.arange(ct)[:, None]  # [ct, 1]
            l_idx = np.arange(L)[None, :]   # [1, L]
            for k in range(K):
                presence[t_idx, l_idx, chunk[:, :, k]] = 1

            # Reshape to [ct * n_experts, L] — one row per (token, expert)
            flat = presence.transpose(0, 2, 1).reshape(-1, L)

            # Pad with 0 on both sides so every run of 1s has a
            # clear start (+1 in diff) and end (-1 in diff).
            padded = np.pad(flat, ((0, 0), (1, 1)), constant_values=0)
            d = np.diff(padded, axis=1)     # shape [ct * n_experts, L+1]

            starts_c = np.where(d == 1)[1]
            ends_c = np.where(d == -1)[1]
            runs = ends_c - starts_c        # run lengths
            if len(runs) > 0:
                all_runs.append(runs)

    if not all_runs:
        return np.array([0], dtype=np.int64)
    all_runs_arr = np.concatenate(all_runs)
    if len(all_runs_arr) == 0:
        return np.array([0], dtype=np.int64)
    max_run = int(all_runs_arr.max())
    counts = np.zeros(max_run + 1, dtype=np.int64)
    np.add.at(counts, all_runs_arr, 1)
    return counts


# %%
# ── Determine n_experts for the continuity analysis ──────────────────────────
all_arrays = [a for arrs in by_dataset.values() for a in arrs]

if not all_arrays:
    print("No data loaded – skipping continuity analysis.")
else:
    n_experts_cont = model_meta["n_routed_experts"]
    if n_experts_cont is None:
        n_experts_cont = max(a.max() for a in all_arrays) + 1
        print(f"(n_routed_experts not in log — inferred {n_experts_cont} from data)")

    print(f"Computing expert continuity runs (n_experts={n_experts_cont}) …")

    # Per-dataset run-length counts
    dataset_run_counts: dict[str, np.ndarray] = {}
    for name, arrays in by_dataset.items():
        rc = compute_continuity_runs(arrays, n_experts_cont)
        dataset_run_counts[name] = rc
        total_runs = rc.sum()
        max_run = len(rc) - 1
        mean_run = (np.arange(len(rc)) * rc).sum() / total_runs if total_runs > 0 else 0
        print(f"  {name:20s}: {total_runs:10,} runs, "
              f"max_run_length={max_run}, mean={mean_run:.2f}")

    # Combined across all datasets
    max_len = max(len(rc) for rc in dataset_run_counts.values())
    combined_counts = np.zeros(max_len, dtype=np.int64)
    for rc in dataset_run_counts.values():
        combined_counts[:len(rc)] += rc
    total_combined = combined_counts.sum()
    mean_combined = (
        (np.arange(len(combined_counts)) * combined_counts).sum() / total_combined
        if total_combined > 0 else 0
    )
    print(f"  {'ALL':20s}: {total_combined:10,} runs, "
          f"max_run_length={len(combined_counts)-1}, mean={mean_combined:.2f}")
    print()

    # ── PMF of run lengths ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, rc in dataset_run_counts.items():
        if rc.sum() == 0:
            continue
        run_ids = np.arange(len(rc))
        pmf = rc / rc.sum()
        ax.plot(run_ids[1:], pmf[1:], marker="o", markersize=4,
                label=name, linewidth=1.3, alpha=0.85)

    # Combined
    if combined_counts.sum() > 0:
        run_ids = np.arange(len(combined_counts))
        pmf = combined_counts / combined_counts.sum()
        ax.plot(run_ids[1:], pmf[1:], marker="s", markersize=5,
                label="All datasets", linewidth=2, color="black", alpha=0.9)

    ax.set_xlabel("Consecutive-layer run length", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title(f"PMF of Expert Continuity Run Lengths — {MODEL_NAME}", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_continuity_pmf.png", dpi=150)
    plt.show()

    # ── Log-scale version for tail visibility ────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, rc in dataset_run_counts.items():
        if rc.sum() == 0:
            continue
        run_ids = np.arange(len(rc))
        pmf = rc / rc.sum()
        mask = pmf[1:] > 0
        ax.plot(run_ids[1:][mask], pmf[1:][mask], marker="o", markersize=4,
                label=name, linewidth=1.3, alpha=0.85)

    if combined_counts.sum() > 0:
        run_ids = np.arange(len(combined_counts))
        pmf = combined_counts / combined_counts.sum()
        mask = pmf[1:] > 0
        ax.plot(run_ids[1:][mask], pmf[1:][mask], marker="s", markersize=5,
                label="All datasets", linewidth=2, color="black", alpha=0.9)

    ax.set_xlabel("Consecutive-layer run length", fontsize=12)
    ax.set_ylabel("Probability (log scale)", fontsize=12)
    ax.set_title(f"PMF of Expert Continuity Run Lengths (log) — {MODEL_NAME}",
                 fontsize=13)
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(left=1)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_continuity_pmf_log.png", dpi=150)
    plt.show()

    print(f"Continuity figures saved to {OUT_DIR}/")
