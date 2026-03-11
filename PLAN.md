# Expert Measurement Benchmark Plan

## Goal

Build a benchmark script in `expert_measurement/` that compares:
1. **Triton fused MoE kernel** (`TritonExperts` via the modular kernel interface)
2. **Native PyTorch per-expert approach** (sequential `torch.matmul` per expert, the reference implementation from `tests/kernels/utils.py:torch_experts`)

on an **A30 GPU (SM80, Ampere)** with controllable expert skewness and model configs.

---

## Important: A30 Constraints

The A30 has compute capability 8.0. This eliminates several kernel variants:

| Variant | A30 Support | Reason |
|---------|-------------|--------|
| TritonExperts (BF16/FP16) | YES | Works on any CUDA device |
| TritonExperts (FP8) | NO | Requires SM89+ (`has_device_capability((8, 9))`) |
| DeepGemmExperts | NO | Requires SM90+ (Hopper) |
| CUTLASS FP8 | NO | Requires SM89+ |
| FlashInfer | NO | Requires SM90+ |

**Therefore, this benchmark will compare Triton vs Native in BF16/FP16 only** — the only precision both approaches support on A30.

---

## File Structure

```
expert_measurement/
├── __init__.py
├── benchmark.py          # Main benchmark entry point
├── expert_distribution.py # Skewness-controlled expert routing
├── model_configs.py       # Load MoE configs from HF model names
└── README.md              # Usage instructions (only if requested)
```

## Detailed Design

### 1. `model_configs.py` — Load MoE Hyperparameters from Model Configs

**What it does**: Given a HuggingFace model name (or a local config.json path), extract the MoE-relevant hyperparameters without downloading weights.

**Implementation**:
- Use `transformers.AutoConfig.from_pretrained(model_name)` to load config (downloads only config.json, ~1KB)
- Extract fields with fallback across naming conventions:
  - `num_experts`: try `num_local_experts`, `n_routed_experts`, `num_experts`, `moe_num_experts`
  - `top_k`: try `num_experts_per_tok`, `moe_top_k`, `moe_topk`
  - `hidden_size`: standard field
  - `intermediate_size`: try `moe_intermediate_size`, `intermediate_size`
- Return a dataclass `MoEModelConfig(num_experts, top_k, hidden_size, intermediate_size, model_name)`
- Also support manual override via CLI args
- Include a few hardcoded presets for offline use (Mixtral-8x7B, DeepSeek-V2-Lite, Qwen2-57B-A14B, etc.)

**Preset configs** (for convenience without network):
```python
PRESETS = {
    "mixtral-8x7b": MoEModelConfig(num_experts=8, top_k=2, hidden_size=4096, intermediate_size=14336),
    "deepseek-v2-lite": MoEModelConfig(num_experts=64, top_k=6, hidden_size=2048, intermediate_size=1408),
    "deepseek-v3": MoEModelConfig(num_experts=256, top_k=8, hidden_size=7168, intermediate_size=2048),
    "qwen2-57b-a14b": MoEModelConfig(num_experts=64, top_k=8, hidden_size=3584, intermediate_size=2560),
}
```

### 2. `expert_distribution.py` — Controllable Expert Routing

**What it does**: Generate `topk_ids` and `topk_weights` with controllable skewness, instead of using random gating.

**Skewness control strategies**:

1. **Uniform**: Each expert gets roughly equal tokens (baseline)
2. **Zipf(alpha)**: Expert popularity follows Zipf's law — expert `i` gets probability proportional to `1/(i+1)^alpha`
   - `alpha=0`: uniform
   - `alpha=1`: moderate skew (80/20-like)
   - `alpha=2`: extreme skew (one expert dominates)
3. **Single-hot**: All tokens go to one expert (worst case for padding waste)
4. **Custom counts**: User provides explicit per-expert token counts

**Implementation**:
```python
def generate_expert_assignments(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    distribution: str = "uniform",  # "uniform", "zipf", "single_hot", "custom"
    zipf_alpha: float = 1.0,
    hot_expert_id: int = 0,
    custom_counts: list[int] | None = None,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (topk_ids: [M, top_k], topk_weights: [M, top_k])"""
```

- For Zipf: compute probability per expert, sample `top_k` experts per token from the distribution (without replacement within a token)
- `topk_weights` are normalized softmax scores (random magnitudes, but normalized per row)
- Returns tensors on CUDA, int32 ids and float32 weights

**Also provide**: a utility function to print the actual expert load distribution for verification:
```python
def print_distribution_stats(topk_ids, num_experts):
    """Print tokens per expert, min/max/std, Gini coefficient."""
```

### 3. `benchmark.py` — Main Benchmark Script

**What it does**: Run Triton fused kernel vs native PyTorch, measure latency, report results.

**Two approaches to benchmark**:

#### Approach A: Triton Fused MoE (from vLLM)
Import and call the kernel directly from vLLM:
```python
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe import fused_topk
```

Or via the modular kernel interface:
```python
from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEModularKernel
from vllm.model_executor.layers.fused_moe.prepare_finalize import MoEPrepareAndFinalizeNoEP
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEQuantConfig
```

Following patterns from `tests/kernels/moe/utils.py:modular_triton_fused_moe()`.

#### Approach B: Native PyTorch (per-expert torch.matmul)
Port the reference implementation from `tests/kernels/utils.py:torch_experts()`:
```python
for i in range(num_experts):
    mask = topk_ids == i
    if mask.sum():
        tmp1 = a[mask] @ w1[i].transpose(0, 1)
        tmp2 = SiluAndMul()(tmp1)
        out[mask] = tmp2 @ w2[i].transpose(0, 1)
```

**Benchmark loop**:
1. Create random weights: `w1: [E, 2*N, K]`, `w2: [E, K, N]` (the `2*N` is for gate+up in SiLU-and-mul)
2. Create random hidden states: `[M, K]`
3. Generate expert assignments using `expert_distribution.py`
4. Warmup (10 iterations)
5. Timed runs (100 iterations), measure with `torch.cuda.Event` for accurate GPU timing
6. Report: mean, median, min, max, std of latency per call

**CLI interface**:
```bash
# Use preset model config
python expert_measurement/benchmark.py --model mixtral-8x7b --num-tokens 128 256 512 1024 --distribution zipf --zipf-alpha 0.0 0.5 1.0 2.0

# Use HF model name (downloads config only)
python expert_measurement/benchmark.py --hf-model mistralai/Mixtral-8x7B-v0.1 --num-tokens 512

# Manual config
python expert_measurement/benchmark.py --num-experts 64 --top-k 6 --hidden-size 2048 --intermediate-size 1408 --num-tokens 32 128 512 --distribution uniform zipf single_hot

# Full sweep
python expert_measurement/benchmark.py --model mixtral-8x7b deepseek-v2-lite --num-tokens 1 32 128 512 1024 --distribution uniform zipf --zipf-alpha 0.0 1.0 2.0 --dtype bfloat16
```

**Output format**: Print a table and optionally write CSV:
```
Model: mixtral-8x7b (E=8, top_k=2, H=4096, N=14336)
Dtype: bfloat16
Distribution: zipf(alpha=1.0)
Tokens per expert: [412, 206, 137, 103, 82, 68, 58, 52] (of 512 tokens * top_k=2)

  Tokens |  Triton (ms) | Native (ms) | Speedup |  Padding Waste
  -------|------------- |-------------|---------|---------------
      32 |        0.15  |       0.22  |  1.47x  |  35.2%
     128 |        0.42  |       1.05  |  2.50x  |  12.1%
     512 |        1.23  |       4.15  |  3.37x  |   3.8%
    1024 |        2.31  |       8.20  |  3.55x  |   1.9%
```

**Padding waste calculation**: For Triton, compute:
```python
# From moe_align_block_size: each expert's tokens rounded up to BLOCK_SIZE_M
padded = sum(round_up(tokens_per_expert[i], block_size) for i in range(E))
actual = sum(tokens_per_expert)
waste = (padded - actual) / padded
```

---

## Potential Discrepancies vs Real Performance

These are the key ways this benchmark may deviate from actual production MoE inference:

### 1. **Missing routing overhead**
Real MoE includes the router linear layer (`hidden_states @ gate_weight`) and `fused_topk()` softmax+topk. Our benchmark skips this since we pre-generate expert assignments. Impact: small (router is cheap relative to expert computation).

### 2. **No memory pressure from KV cache / attention**
In real inference, GPU memory is heavily used by KV cache, attention buffers, and model weights across all layers. This creates L2 cache pressure that slows both approaches. Our benchmark has minimal cache pressure since we only have MoE weights. Impact: **the benchmark overestimates throughput for both approaches**, but more so for the native approach (which does many small matmuls that benefit more from cache hits).

### 3. **Triton autotuning state**
The Triton kernel uses autotuned tile sizes (`BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`). In production vLLM, these are tuned per-model and cached. In our benchmark, Triton will autotune on first run. Impact: first run is slower; subsequent runs match production if shapes match.

### 4. **CUDA graph effects**
Production vLLM captures the MoE kernel in CUDA graphs (for decode). CUDA graphs eliminate launch overhead. Our benchmark uses eager execution. Impact: **benchmark slightly overestimates Triton launch overhead** relative to production.

### 5. **Tensor parallelism (TP) slicing**
In production with TP>1, `intermediate_size` is divided across GPUs. A TP=2 Mixtral has `N=14336/2=7168` per GPU. Our benchmark uses the full unsharded size by default. Users should pass `--tp` to simulate sharding. Impact: smaller N changes the compute/memory ratio and can shift which tile sizes are optimal.

### 6. **Weight data is random, not real**
Random weights have different numerical distribution than trained weights. This doesn't affect latency (GPU doesn't branch on values), but affects accuracy validation if we compare outputs. Impact: **none for latency benchmarking**.

### 7. **Expert distribution in practice**
Real models show moderate skew that varies per-layer and per-batch. Our synthetic distributions (uniform, Zipf) are simplified. In particular:
- Real routing rarely sends ALL tokens to one expert
- Real routing has correlations across layers
- Load balancing losses in training push toward uniformity
Impact: Zipf with alpha=0.5–1.0 is a reasonable proxy for real skew.

### 8. **Padding granularity difference: BLOCK_SIZE_M**
Triton's `moe_align_block_size` pads to `BLOCK_SIZE_M` which is autotuned (typically 16, 32, 64, or 128). The benchmark will use whatever Triton selects. This is accurate for Triton, but note that DeepGEMM (not runnable on A30) always pads to 128 — so the A30 benchmark cannot measure that larger padding cost.

### 9. **Native approach uses cuBLAS, not raw CUDA**
The "native" `torch.matmul` approach calls cuBLAS under the hood. cuBLAS is highly optimized per-shape. For large per-expert batch sizes, cuBLAS can be very fast. The native approach's weakness is **Python loop overhead and many small kernel launches** (one matmul per expert), not the matmul quality itself.

### 10. **torch.compile effects**
If `torch.compile` is enabled (not default in our benchmark), the native per-expert loop could be fused/optimized by the compiler. Our benchmark tests raw eager execution. Impact: the native approach could be faster with compile, but this is not how vLLM uses it in production.

---

## Implementation Notes

- **Import directly from vLLM source**: `sys.path.insert(0, vllm_root)` and import `fused_experts`, `TritonExperts`, etc. No copying kernel code.
- **GPU timing**: Use `torch.cuda.Event(enable_timing=True)` with `synchronize()` for accurate measurement.
- **Warmup**: 10 iterations before timing to trigger JIT compilation and autotuning.
- **Memory cleanup**: `torch.cuda.empty_cache()` between different configurations to avoid OOM.
- **Seed control**: Fixed seed for reproducible expert distributions.
