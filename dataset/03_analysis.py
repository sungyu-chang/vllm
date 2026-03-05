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
#
# ## Cache behaviour
# On first run the script analyses the log file and writes all results to
# CSV / JSON under `figures_{model}/data/`.  On subsequent runs it detects
# the cached data and skips the (slow) analysis step, going straight to
# figure generation.  Delete `figures_{model}/data/.data_ready` (or the
# whole `data/` folder) to force a re-analysis.

# %%
import base64
import csv
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

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_NAME = "deepseek"      # ← set your model name here
# Override from command line:  python 03_analysis.py <model_name>
if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    MODEL_NAME = sys.argv[1]

LOG_FILE = f"expert_log_{MODEL_NAME}.jsonl"
LAYER_ID = 0                      # change to inspect a different layer (3-2)

OUT_DIR  = Path(f"figures_{MODEL_NAME}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = OUT_DIR / "data"       # CSV / JSON cache lives here
DATA_DIR.mkdir(parents=True, exist_ok=True)
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
        fig.tight_layout()
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

    fig.tight_layout()


print(f"Model:      {MODEL_NAME}")
print(f"Log file:   {LOG_FILE}")
print(f"Output dir: {OUT_DIR}/")
print(f"Cache dir:  {DATA_DIR}/")
print()

# %% [markdown]
# ## Helpers: decode, load, count, KL, continuity

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


def load_log(log_path: str) -> tuple[dict[str, list[np.ndarray]], dict]:
    """Read expert_log.jsonl and group expert arrays by dataset name."""
    by_dataset: dict[str, list[np.ndarray]] = defaultdict(list)
    model_meta: dict = {"n_routed_experts": None, "n_shared_experts": 0}
    skipped = 0

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if model_meta["n_routed_experts"] is None and rec.get("n_routed_experts"):
                model_meta["n_routed_experts"] = rec["n_routed_experts"]
                model_meta["n_shared_experts"] = rec.get("n_shared_experts", 0) or 0
            arr = decode_experts(rec)
            if arr is None:
                skipped += 1
                continue
            by_dataset[rec["dataset_name"]].append(arr)

    print("Loaded records per dataset:")
    for name, arrays in by_dataset.items():
        total_tok = sum(a.shape[0] for a in arrays)
        print(f"  {name:20s}: {len(arrays):5d} samples, {total_tok:8d} total tokens")
    if skipped:
        print(f"  (skipped {skipped} records with no expert data)")

    n_r = model_meta["n_routed_experts"]
    n_s = model_meta["n_shared_experts"]
    print(f"\nModel expert config: n_routed_experts={n_r}, n_shared_experts={n_s}")
    if n_s:
        print(f"  NOTE: {n_s} shared expert(s) always active — not in logged arrays.")

    return dict(by_dataset), model_meta


def expert_counts_all_layers(arrays: list[np.ndarray]) -> np.ndarray:
    """Aggregate expert selection counts across ALL tokens and ALL layers."""
    if not arrays:
        return np.array([], dtype=np.int64)
    max_expert = max(a.max() for a in arrays)
    counts = np.zeros(max_expert + 1, dtype=np.int64)
    for a in arrays:
        np.add.at(counts, a.ravel(), 1)
    return counts


def expert_counts_one_layer(arrays: list[np.ndarray], layer_id: int) -> np.ndarray:
    """Aggregate expert selection counts for a SINGLE layer."""
    if not arrays:
        return np.array([], dtype=np.int64)
    valid = [a for a in arrays if a.shape[1] > layer_id]
    if not valid:
        return np.array([], dtype=np.int64)
    max_expert = max(a[:, layer_id, :].max() for a in valid)
    counts = np.zeros(max_expert + 1, dtype=np.int64)
    for a in valid:
        np.add.at(counts, a[:, layer_id, :].ravel(), 1)
    return counts


def counts_to_cdf(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a count array into a CDF over expert IDs."""
    if len(counts) == 0:
        return np.array([]), np.array([])
    return np.arange(len(counts)), np.cumsum(counts) / counts.sum()


def counts_to_pmf(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a count array into a PMF."""
    if len(counts) == 0:
        return np.array([]), np.array([])
    return np.arange(len(counts)), counts / counts.sum()


def add_shared_expert_footnote(fig, n_shared: int) -> None:
    if n_shared <= 0:
        return
    note = (f"Note: {n_shared} shared expert(s) always active on every token "
            f"— not included in these counts (routed experts only).")
    fig.text(0.01, 0.01, note, fontsize=8, color="dimgray",
             va="bottom", ha="left", style="italic")
    fig.subplots_adjust(bottom=0.12)


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-12) -> float:
    """KL(P ‖ Q) with additive smoothing."""
    p = np.asarray(p, dtype=np.float64) + epsilon
    q = np.asarray(q, dtype=np.float64) + epsilon
    p = p / p.sum(); q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def symmetric_kl(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-12) -> float:
    return (kl_divergence(p, q, epsilon) + kl_divergence(q, p, epsilon)) / 2.0


def pad_to_same_length(*arrays: np.ndarray) -> list[np.ndarray]:
    max_len = max(len(a) for a in arrays)
    return [np.pad(a, (0, max_len - len(a))) if len(a) < max_len else a
            for a in arrays]


def compute_continuity_runs(arrays: list[np.ndarray], n_experts: int) -> np.ndarray:
    """Count consecutive-layer run lengths aggregated over whole prompts.

    For each sample (prompt + output tokens), an expert is considered
    'present' at a layer if it was selected by ANY token in that sample
    at that layer.  A run of length r means the expert stayed present for
    r consecutive layers across the entire prompt.

    Args:
        arrays:    list of [total_tokens, num_layers, topk] int32 arrays
        n_experts: number of routed experts

    Returns:
        1-D int64 array where counts[r] = number of runs of length r.
        counts[0] is always 0.
    """
    all_runs: list[np.ndarray] = []
    l_idx = None  # initialised on first array

    for arr in arrays:
        T, L, K = arr.shape
        if l_idx is None or l_idx.shape[1] != L:
            l_idx = np.arange(L)[None, :]   # [1, L] — broadcasts with arr[:, :, k] [T, L]

        # present[l, e] = 1 iff expert e was selected by ANY token at layer l
        present = np.zeros((L, n_experts), dtype=np.int8)
        for k in range(K):
            present[l_idx, arr[:, :, k]] = 1   # arr[:, :, k] is [T, L]

        # One row per expert: sequence of 0/1 over layers
        flat   = present.T                                          # [n_experts, L]
        padded = np.pad(flat, ((0, 0), (1, 1)), constant_values=0)
        d      = np.diff(padded, axis=1)
        starts_c = np.where(d == 1)[1]
        ends_c   = np.where(d == -1)[1]
        runs = ends_c - starts_c
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

# %% [markdown]
# ## 1. Analysis Phase — compute results or load from cache
#
# All computed results are written to `OUT_DIR/data/` as CSV / JSON.
# If the sentinel file `.data_ready` is present the slow analysis step
# is skipped and figures are redrawn directly from the cached files.

# %%
_READY = DATA_DIR / ".data_ready"

# In-memory data structures (populated in either branch below)
expert_counts_all:       dict[str, np.ndarray] = {}  # name → 1-D int64 counts (all layers)
expert_counts_per_layer: dict[str, list]       = {}  # name → list[np.ndarray] indexed by layer
kl_overall_rows:  list[tuple] = []   # (ds_a, ds_b, skl, flag_str)
kl_matrix:        np.ndarray | None = None  # [n_pairs, n_layers]
kl_pairs:         list[tuple] = []          # [(ds_a, ds_b), …]
dataset_run_counts: dict[str, np.ndarray] = {}
combined_counts:    np.ndarray | None = None
dataset_names: list[str] = []
n_experts:     int = 0
N_SHARED:      int = 0
num_layers:    int = 0


# ── CSV / JSON save helpers ──────────────────────────────────────────────────
def _save_analysis() -> None:
    """Write all analysis results to DATA_DIR."""

    # 1. expert_counts_all_layers.csv
    max_e = max((len(v) for v in expert_counts_all.values()), default=0)
    with open(DATA_DIR / "expert_counts_all_layers.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["expert_id"] + dataset_names)
        for e in range(max_e):
            row = [e] + [int(expert_counts_all[n][e]) if e < len(expert_counts_all[n]) else 0
                         for n in dataset_names]
            w.writerow(row)

    # 2. expert_counts_per_layer.csv
    with open(DATA_DIR / "expert_counts_per_layer.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "expert_id"] + dataset_names)
        for layer in range(num_layers):
            max_e_l = max((len(expert_counts_per_layer[n][layer]) for n in dataset_names), default=0)
            for e in range(max_e_l):
                row = [layer, e] + [
                    int(expert_counts_per_layer[n][layer][e])
                    if e < len(expert_counts_per_layer[n][layer]) else 0
                    for n in dataset_names
                ]
                w.writerow(row)

    # 3. kl_overall.csv
    with open(DATA_DIR / "kl_overall.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset_a", "dataset_b", "skl", "flag"])
        for row in kl_overall_rows:
            w.writerow(row)

    # 4. kl_per_layer.csv
    if kl_matrix is not None and kl_pairs:
        with open(DATA_DIR / "kl_per_layer.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset_a", "dataset_b", "layer", "skl"])
            for i, (ds_a, ds_b) in enumerate(kl_pairs):
                for layer in range(kl_matrix.shape[1]):
                    w.writerow([ds_a, ds_b, layer, float(kl_matrix[i, layer])])

    # 5. continuity_runs.csv
    max_r = max((len(rc) for rc in dataset_run_counts.values()), default=0)
    if combined_counts is not None:
        max_r = max(max_r, len(combined_counts))
    with open(DATA_DIR / "continuity_runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_length"] + dataset_names + ["combined"])
        for r in range(max_r):
            row = [r]
            for n in dataset_names:
                rc = dataset_run_counts.get(n, np.array([0]))
                row.append(int(rc[r]) if r < len(rc) else 0)
            row.append(int(combined_counts[r])
                       if combined_counts is not None and r < len(combined_counts) else 0)
            w.writerow(row)

    # 6. model_meta.json
    with open(DATA_DIR / "model_meta.json", "w") as f:
        json.dump({
            "n_routed_experts": n_experts,
            "n_shared_experts": N_SHARED,
            "num_layers":       num_layers,
            "dataset_names":    dataset_names,
        }, f, indent=2)

    print(f"Analysis results cached to {DATA_DIR}/")


def _load_analysis() -> None:
    """Load all analysis results from DATA_DIR into the global data structures."""
    global expert_counts_all, expert_counts_per_layer, kl_overall_rows
    global kl_matrix, kl_pairs, dataset_run_counts, combined_counts
    global dataset_names, n_experts, N_SHARED, num_layers

    with open(DATA_DIR / "model_meta.json") as f:
        meta = json.load(f)
    dataset_names = meta["dataset_names"]
    num_layers    = meta["num_layers"]
    n_experts     = meta["n_routed_experts"]
    N_SHARED      = meta["n_shared_experts"]

    # expert_counts_all_layers.csv
    cols: dict[str, list] = defaultdict(list)
    with open(DATA_DIR / "expert_counts_all_layers.csv", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        col_names = header[1:]
        for row in r:
            for i, name in enumerate(col_names):
                cols[name].append(int(row[i + 1]))
    expert_counts_all = {name: np.array(cols[name], dtype=np.int64) for name in col_names}

    # expert_counts_per_layer.csv
    per_layer_raw: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    with open(DATA_DIR / "expert_counts_per_layer.csv", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        col_names2 = header[2:]
        for row in r:
            layer = int(row[0])
            for i, name in enumerate(col_names2):
                per_layer_raw[layer][name].append(int(row[2 + i]))
    expert_counts_per_layer = {name: [] for name in col_names2}
    for layer in range(num_layers):
        for name in col_names2:
            expert_counts_per_layer[name].append(
                np.array(per_layer_raw[layer][name], dtype=np.int64))

    # kl_overall.csv
    kl_overall_rows = []
    with open(DATA_DIR / "kl_overall.csv", newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            kl_overall_rows.append((row[0], row[1], float(row[2]), row[3]))

    # kl_per_layer.csv
    kl_path = DATA_DIR / "kl_per_layer.csv"
    if kl_path.exists():
        pair_idx: dict[tuple, int] = {}
        rows_raw = []
        with open(kl_path, newline="") as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                ds_a, ds_b, layer, skl = row[0], row[1], int(row[2]), float(row[3])
                key = (ds_a, ds_b)
                if key not in pair_idx:
                    pair_idx[key] = len(pair_idx)
                    kl_pairs.append(key)
                rows_raw.append((pair_idx[key], layer, skl))
        if rows_raw:
            kl_matrix = np.zeros((len(kl_pairs), num_layers))
            for pi, li, skl in rows_raw:
                kl_matrix[pi, li] = skl

    # continuity_runs.csv
    rc_cols: dict[str, list] = defaultdict(list)
    combined_list: list[int] = []
    with open(DATA_DIR / "continuity_runs.csv", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        rc_names = header[1:-1]   # strip "run_length" and "combined"
        for row in r:
            for i, name in enumerate(rc_names):
                rc_cols[name].append(int(row[1 + i]))
            combined_list.append(int(row[-1]))
    dataset_run_counts = {name: np.array(rc_cols[name], dtype=np.int64) for name in rc_names}
    combined_counts = np.array(combined_list, dtype=np.int64)


# ── Main cache-conditional block ─────────────────────────────────────────────
if _READY.exists():
    print(f"Cached analysis found in {DATA_DIR}/ — loading from CSV …")
    _load_analysis()
    print(f"  Datasets   : {dataset_names}")
    print(f"  Layers     : {num_layers}")
    print(f"  n_experts  : {n_experts}  (shared: {N_SHARED})")
    print()

else:
    print("Running analysis (this may take a while for large logs) …")
    print()

    by_dataset, model_meta = load_log(LOG_FILE)
    N_SHARED      = model_meta["n_shared_experts"]
    dataset_names = list(by_dataset.keys())
    all_arrays    = [a for arrs in by_dataset.values() for a in arrs]
    num_layers    = max(a.shape[1] for a in all_arrays) if all_arrays else 0
    n_experts     = (model_meta["n_routed_experts"] or
                     (int(max(a.max() for a in all_arrays)) + 1 if all_arrays else 0))
    n_experts_cont = n_experts   # used for continuity presence matrix size

    # ── Expert counts ────────────────────────────────────────────────────
    print("Computing expert counts …")
    for name, arrays in by_dataset.items():
        expert_counts_all[name] = expert_counts_all_layers(arrays)

    for name, arrays in by_dataset.items():
        expert_counts_per_layer[name] = [
            expert_counts_one_layer(arrays, layer)
            for layer in range(num_layers)
        ]

    # ── KL divergence ────────────────────────────────────────────────────
    kl_pairs = list(combinations(dataset_names, 2))
    if kl_pairs:
        print("Computing KL divergences …")
        pmfs_overall = {}
        for name in dataset_names:
            _, pmf = counts_to_pmf(expert_counts_all[name])
            pmfs_overall[name] = pmf

        for ds_a, ds_b in kl_pairs:
            pa, pb = pad_to_same_length(pmfs_overall[ds_a], pmfs_overall[ds_b])
            skl = symmetric_kl(pa, pb)
            flag = "significant" if skl >= 0.1 else ("moderate" if skl >= 0.01 else "")
            kl_overall_rows.append((ds_a, ds_b, skl, flag))

        kl_matrix = np.zeros((len(kl_pairs), num_layers))
        for i, (ds_a, ds_b) in enumerate(kl_pairs):
            for layer in range(num_layers):
                ca = expert_counts_per_layer[ds_a][layer]
                cb = expert_counts_per_layer[ds_b][layer]
                _, pa = counts_to_pmf(ca)
                _, pb = counts_to_pmf(cb)
                if len(pa) > 0 and len(pb) > 0:
                    pa, pb = pad_to_same_length(pa, pb)
                    kl_matrix[i, layer] = symmetric_kl(pa, pb)

    # ── Expert continuity ─────────────────────────────────────────────────
    if all_arrays:
        print(f"Computing expert continuity runs (n_experts={n_experts_cont}) …")
        for name, arrays in by_dataset.items():
            rc = compute_continuity_runs(arrays, n_experts_cont)
            dataset_run_counts[name] = rc
            total_r = rc.sum()
            mean_r  = (np.arange(len(rc)) * rc).sum() / total_r if total_r > 0 else 0
            print(f"  {name:20s}: {total_r:10,} runs, "
                  f"max={len(rc)-1}, mean={mean_r:.2f}")

        max_r = max(len(rc) for rc in dataset_run_counts.values())
        combined_counts = np.zeros(max_r, dtype=np.int64)
        for rc in dataset_run_counts.values():
            combined_counts[:len(rc)] += rc

    # ── Save cache ────────────────────────────────────────────────────────
    _save_analysis()
    _READY.touch()
    print()

# %% [markdown]
# ## 2. KL Divergence Summary
#
# Symmetric KL divergence: SKL(P, Q) = (KL(P‖Q) + KL(Q‖P)) / 2.
# Thresholds: ≥ 0.1 significant  |  ≥ 0.01 moderate.

# %%
if len(kl_overall_rows) > 0:
    print("=" * 65)
    print("Pairwise Symmetric KL Divergence  (all layers aggregated)")
    print("=" * 65)
    for ds_a, ds_b, skl, flag in kl_overall_rows:
        suffix = f"  *** {flag}" if flag else ""
        print(f"  {ds_a:20s} vs {ds_b:20s}:  SKL = {skl:.6f}{suffix}")
    print()

    if kl_matrix is not None:
        print("Per-layer Symmetric KL Divergence")
        print("-" * 65)
        for i, (ds_a, ds_b) in enumerate(kl_pairs):
            row = kl_matrix[i]
            sig_layers = np.where(row >= 0.1)[0]
            mod_layers = np.where((row >= 0.01) & (row < 0.1))[0]
            print(f"  {ds_a} vs {ds_b}:")
            print(f"    Mean SKL = {row.mean():.6f},  "
                  f"Max SKL = {row.max():.6f} (layer {row.argmax()})")
            if len(sig_layers) > 0:
                print(f"    Significant layers (SKL >= 0.1): {sig_layers.tolist()}")
            if len(mod_layers) > 0:
                print(f"    Moderate layers  (0.01 <= SKL < 0.1): {mod_layers.tolist()}")
        print()
else:
    print("(Only one dataset — skipping pairwise KL comparison.)\n")

# %% [markdown]
# ## 3-1. Aggregated CDF of expert choice across all layers

# %%
fig, ax = plt.subplots(figsize=(9, 5))

for name in dataset_names:
    expert_ids, cdf = counts_to_cdf(expert_counts_all[name])
    if len(expert_ids) == 0:
        continue
    ax.plot(expert_ids, cdf, label=name, linewidth=1.5)

ax.set_xlabel("Expert ID (routed experts only)")
ax.set_ylabel("Cumulative selection probability")
ax.set_title("Aggregated CDF of Expert Selection (all layers)")
ax.legend(loc="lower right")
ax.set_xlim(left=0)
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
add_shared_expert_footnote(fig, N_SHARED)
style_fig(fig)
fig.savefig(OUT_DIR / "fig_cdf_all_layers.png", dpi=150)
plt.show()

# %% [markdown]
# ## 3-2. CDF of expert choice at a specific layer (change `LAYER_ID` at the top)

# %%
fig, ax = plt.subplots(figsize=(9, 5))

for name in dataset_names:
    counts    = expert_counts_per_layer[name][LAYER_ID]
    expert_ids, cdf = counts_to_cdf(counts)
    if len(expert_ids) == 0:
        continue
    ax.plot(expert_ids, cdf, label=name, linewidth=1.5)

ax.set_xlabel("Expert ID (routed experts only)")
ax.set_ylabel("Cumulative selection probability")
ax.set_title(f"Aggregated CDF of Expert Selection at Layer {LAYER_ID}")
ax.legend(loc="lower right")
ax.set_xlim(left=0)
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
add_shared_expert_footnote(fig, N_SHARED)
style_fig(fig)
fig.savefig(OUT_DIR / f"fig_cdf_layer{LAYER_ID}.png", dpi=150)
plt.show()

# %% [markdown]
# ## 4-1. PMF of expert selection — all layers

# %%
fig, ax = plt.subplots(figsize=(9, 5))

for name in dataset_names:
    expert_ids, pmf = counts_to_pmf(expert_counts_all[name])
    if len(expert_ids) == 0:
        continue
    ax.plot(expert_ids, pmf, label=name, linewidth=1.2, alpha=0.85)

if dataset_names:
    n_e = max(len(expert_counts_all[n]) for n in dataset_names)
    ax.axhline(1 / n_e, color="black", linestyle=":", linewidth=1, label="Uniform (routed)")

ax.set_xlabel("Expert ID (routed experts only)")
ax.set_ylabel("Selection probability")
ax.set_title("PMF of Expert Selection (all layers)")
ax.legend(loc="upper right")
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
add_shared_expert_footnote(fig, N_SHARED)
style_fig(fig)
fig.savefig(OUT_DIR / "fig_pmf_all_layers.png", dpi=150)
plt.show()

# %% [markdown]
# ## 4-2. PMF of expert selection at a specific layer

# %%
fig, ax = plt.subplots(figsize=(9, 5))

for name in dataset_names:
    counts = expert_counts_per_layer[name][LAYER_ID]
    expert_ids, pmf = counts_to_pmf(counts)
    if len(expert_ids) == 0:
        continue
    ax.plot(expert_ids, pmf, label=name, linewidth=1.2, alpha=0.85)

if dataset_names:
    n_e = max(len(expert_counts_per_layer[n][LAYER_ID]) for n in dataset_names)
    if n_e > 0:
        ax.axhline(1 / n_e, color="black", linestyle=":", linewidth=1, label="Uniform (routed)")

ax.set_xlabel("Expert ID (routed experts only)")
ax.set_ylabel("Selection probability")
ax.set_title(f"PMF of Expert Selection at Layer {LAYER_ID}")
ax.legend(loc="upper right")
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
add_shared_expert_footnote(fig, N_SHARED)
style_fig(fig)
fig.savefig(OUT_DIR / f"fig_pmf_layer{LAYER_ID}.png", dpi=150)
plt.show()

# %% [markdown]
# ## 4-3. PMF (line) and absolute invocation count (scatter) — all layers

# %%
# ── PMF: line figure ──────────────────────────────────────────────────────────
fig_pmf, ax_pmf = plt.subplots(figsize=(9, 5))

for name in dataset_names:
    expert_ids, pmf = counts_to_pmf(expert_counts_all[name])
    if len(expert_ids) == 0:
        continue
    ax_pmf.plot(expert_ids, pmf, label=name, linewidth=1.2, alpha=0.85)

if dataset_names:
    n_e = max(len(expert_counts_all[n]) for n in dataset_names)
    if n_e > 0:
        ax_pmf.axhline(1 / n_e, color="black", linestyle=":", linewidth=1, label="Uniform")

ax_pmf.set_xlabel("Expert ID")
ax_pmf.set_ylabel("Selection probability")
ax_pmf.set_title("PMF of Expert Selection (all layers)")
ax_pmf.legend(loc="upper right")
ax_pmf.set_xlim(left=0)
ax_pmf.set_ylim(bottom=0)
ax_pmf.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
style_fig(fig_pmf)
fig_pmf.savefig(OUT_DIR / "fig_pmf_all_layers_v2.png", dpi=150)
plt.show()

# ── Absolute count: scatter dot figure ───────────────────────────────────────
fig_cnt, ax_cnt = plt.subplots(figsize=(9, 5))
total_all = 0

for name in dataset_names:
    counts = expert_counts_all[name]
    expert_ids = np.arange(len(counts))
    if len(expert_ids) == 0:
        continue
    total_all += counts.sum()
    ax_cnt.scatter(expert_ids, counts, label=name, s=8, alpha=0.7)

if dataset_names:
    n_e = max(len(expert_counts_all[n]) for n in dataset_names)
    if n_e > 0 and total_all > 0:
        ax_cnt.axhline(total_all / n_e, color="black", linestyle=":", linewidth=1, label="Uniform")

ax_cnt.set_xlabel("Expert ID")
ax_cnt.set_ylabel("Number of invocations")
ax_cnt.set_title("Absolute Invocation Count per Expert (all layers)")
ax_cnt.legend(loc="upper right")
ax_cnt.set_xlim(left=0)
ax_cnt.set_ylim(bottom=0)
ax_cnt.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
style_fig(fig_cnt)
fig_cnt.savefig(OUT_DIR / "fig_count_all_layers.png", dpi=150)
plt.show()

# %% [markdown]
# ## 4-4. Per-layer PMF (line) and absolute count (scatter) — saved to `layer_figures/`

# %%
if not dataset_names:
    print("No data — run 02_run_inference.py first.")
else:
    print(f"Model has {num_layers} MoE layers. Generating per-layer figures …")

    layer_fig_dir = OUT_DIR / "layer_figures"
    layer_fig_dir.mkdir(exist_ok=True)

    for layer in range(num_layers):
        # ── PMF ──────────────────────────────────────────────────────────
        fig_pmf, ax_pmf = plt.subplots(figsize=(9, 5))
        n_e_layer = 0

        for name in dataset_names:
            counts = expert_counts_per_layer[name][layer]
            expert_ids, pmf = counts_to_pmf(counts)
            if len(expert_ids) == 0:
                continue
            n_e_layer = max(n_e_layer, len(counts))
            ax_pmf.plot(expert_ids, pmf, label=name, linewidth=1.2, alpha=0.85)

        if n_e_layer > 0:
            ax_pmf.axhline(1 / n_e_layer, color="black", linestyle=":", linewidth=1, label="Uniform")

        ax_pmf.set_xlabel("Expert ID")
        ax_pmf.set_ylabel("Selection probability")
        ax_pmf.set_title(f"PMF of Expert Selection — Layer {layer}")
        ax_pmf.legend(loc="upper right")
        ax_pmf.set_xlim(left=0)
        ax_pmf.set_ylim(bottom=0)
        ax_pmf.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        style_fig(fig_pmf)
        fig_pmf.savefig(layer_fig_dir / f"pmf_layer{layer:02d}.png", dpi=150)
        plt.close(fig_pmf)

        # ── Count scatter ─────────────────────────────────────────────────
        fig_cnt, ax_cnt = plt.subplots(figsize=(9, 5))
        total_layer = 0

        for name in dataset_names:
            counts = expert_counts_per_layer[name][layer]
            expert_ids = np.arange(len(counts))
            if len(expert_ids) == 0:
                continue
            total_layer += counts.sum()
            ax_cnt.scatter(expert_ids, counts, label=name, s=8, alpha=0.7)

        if n_e_layer > 0 and total_layer > 0:
            ax_cnt.axhline(total_layer / n_e_layer, color="black", linestyle=":", linewidth=1, label="Uniform")

        ax_cnt.set_xlabel("Expert ID")
        ax_cnt.set_ylabel("Number of invocations")
        ax_cnt.set_title(f"Absolute Invocation Count per Expert — Layer {layer}")
        ax_cnt.legend(loc="upper right")
        ax_cnt.set_xlim(left=0)
        ax_cnt.set_ylim(bottom=0)
        ax_cnt.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        style_fig(fig_cnt)
        fig_cnt.savefig(layer_fig_dir / f"count_layer{layer:02d}.png", dpi=150)
        plt.close(fig_cnt)

    print(f"Saved {num_layers * 2} figures (PMF + count per layer) to '{layer_fig_dir}/'")

# %% [markdown]
# ## Bonus: per-layer CDF sweep

# %%
if dataset_names:
    layer_cdf_dir = OUT_DIR / "layer_cdfs"
    layer_cdf_dir.mkdir(exist_ok=True)

    for layer in range(num_layers):
        fig, ax = plt.subplots(figsize=(9, 5))
        for name in dataset_names:
            counts = expert_counts_per_layer[name][layer]
            expert_ids, cdf = counts_to_cdf(counts)
            if len(expert_ids) == 0:
                continue
            ax.plot(expert_ids, cdf, label=name, linewidth=1.5)

        ax.set_xlabel("Expert ID")
        ax.set_ylabel("Cumulative selection probability")
        ax.set_title(f"CDF of Expert Selection — Layer {layer}")
        ax.legend(loc="lower right")
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        style_fig(fig)
        fig.savefig(layer_cdf_dir / f"cdf_layer{layer:02d}.png", dpi=150)
        plt.close(fig)

    print(f"Saved {num_layers} CDF figures to {layer_cdf_dir}/")
else:
    print("No data loaded — run 02_run_inference.py first.")

# %% [markdown]
# ## 5. KL Divergence Heatmap

# %%
if kl_matrix is not None and len(kl_pairs) > 0:
    fig, ax = plt.subplots(
        figsize=(max(10, num_layers * 0.3), max(3, len(kl_pairs) * 0.8 + 1.5)))
    im = ax.imshow(kl_matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Dataset pair")
    ax.set_yticks(range(len(kl_pairs)))
    ax.set_yticklabels([f"{a} vs {b}" for a, b in kl_pairs], fontsize=9)
    ax.set_title(f"Per-layer Symmetric KL Divergence — {MODEL_NAME}")
    fig.colorbar(im, ax=ax, label="Symmetric KL")
    style_fig(fig)
    fig.savefig(OUT_DIR / "fig_kl_per_layer_heatmap.png", dpi=150)
    plt.show()

# %% [markdown]
# ## 6. Expert Continuity Analysis
#
# PMF of consecutive-layer run lengths per dataset and combined.

# %%
if not dataset_run_counts:
    print("No continuity data available.")
else:
    total_combined = combined_counts.sum() if combined_counts is not None else 0
    mean_combined  = (
        (np.arange(len(combined_counts)) * combined_counts).sum() / total_combined
        if total_combined > 0 else 0
    )
    print(f"  {'ALL':20s}: {total_combined:10,} runs, "
          f"max={len(combined_counts)-1}, mean={mean_combined:.2f}")
    print()

    # ── PMF linear scale ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, rc in dataset_run_counts.items():
        if rc.sum() == 0:
            continue
        run_ids = np.arange(len(rc))
        pmf = rc / rc.sum()
        ax.plot(run_ids[1:], pmf[1:], marker="o", markersize=4,
                label=name, linewidth=1.3, alpha=0.85)

    if combined_counts is not None and combined_counts.sum() > 0:
        run_ids = np.arange(len(combined_counts))
        pmf = combined_counts / combined_counts.sum()
        ax.plot(run_ids[1:], pmf[1:], marker="s", markersize=5,
                label="All datasets", linewidth=2, color="black", alpha=0.9)

    ax.set_xlabel("Consecutive-layer run length")
    ax.set_ylabel("Probability")
    ax.set_title(f"PMF of Expert Continuity Run Lengths — {MODEL_NAME}")
    ax.legend(loc="upper right")
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    style_fig(fig)
    fig.savefig(OUT_DIR / "fig_continuity_pmf.png", dpi=150)
    plt.show()

    # ── PMF log scale ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, rc in dataset_run_counts.items():
        if rc.sum() == 0:
            continue
        run_ids = np.arange(len(rc))
        pmf = rc / rc.sum()
        mask = pmf[1:] > 0
        ax.plot(run_ids[1:][mask], pmf[1:][mask], marker="o", markersize=4,
                label=name, linewidth=1.3, alpha=0.85)

    if combined_counts is not None and combined_counts.sum() > 0:
        run_ids = np.arange(len(combined_counts))
        pmf = combined_counts / combined_counts.sum()
        mask = pmf[1:] > 0
        ax.plot(run_ids[1:][mask], pmf[1:][mask], marker="s", markersize=5,
                label="All datasets", linewidth=2, color="black", alpha=0.9)

    ax.set_xlabel("Consecutive-layer run length")
    ax.set_ylabel("Probability (log scale)")
    ax.set_title(f"PMF of Expert Continuity Run Lengths (log) — {MODEL_NAME}")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    ax.set_xlim(left=1)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    style_fig(fig)
    fig.savefig(OUT_DIR / "fig_continuity_pmf_log.png", dpi=150)
    plt.show()

    print(f"Continuity figures saved to {OUT_DIR}/")
