# GPU Optimization Comparison — WaveFollower vs WaveTrader Colab

WaveFollower_Training.ipynb is the gold standard. All new training notebooks must match it.

## Side-by-Side Comparison

| Optimization | WaveFollower_Training.ipynb | WaveTrader_Colab.ipynb | Impact |
|---|---|---|---|
| **cuDNN benchmark** | `torch.backends.cudnn.benchmark = True` | Missing | Auto-tunes conv kernels per input size |
| **TF32 matmul** | `allow_tf32 = True` (cuda + cudnn) | Missing | 2-3x matmul speedup on Ampere+ GPUs |
| **torch.compile** | `mode="reduce-overhead"` | Missing | 15-30% kernel fusion speedup |
| **Mixed precision** | `autocast("cuda")` + `GradScaler` | Missing entirely | Halves VRAM, ~2x matmul throughput |
| **Fused AdamW** | `AdamW(fused=True)` | Missing | Fewer CUDA kernel launches |
| **Gradient zeroing** | `zero_grad(set_to_none=True)` | Standard `zero_grad()` | Avoids memset overhead |
| **non_blocking transfers** | `.to(device, non_blocking=True)` | `.to(device)` | Async CPU→GPU copies |
| **Pre-cached dataset** | `PreCachedMTFDataset` — all tensors in RAM | Raw pandas per `__getitem__` | Eliminates pandas bottleneck |
| **Batch size** | 640 (fills T4 GPU) | 32 (MTF) / 64 (single) | 10-20x more GPU utilization |
| **Learning rate** | 2e-4 (linear-scaled with batch) | 1e-4 (unscaled) | Matches batch size scaling |
| **DataLoader workers** | 4 + `persistent_workers=True` | 2, no persistence | Workers stay alive between epochs |
| **Prefetch factor** | `prefetch_factor=4` | Default (2) | More CPU/GPU overlap |
| **pin_memory** | `True` | `True` | Same (both have this) |
| **drop_last** | `True` on train loader | Missing | Avoids small last-batch stall |
| **Val batch size** | 2x train batch (1280) | Same as train | Uses spare VRAM (no gradients) |
| **Training loop** | Custom inline `train_one_epoch()` | Delegates to `train_model()` | Full control over AMP/clipping |
| **Grad clipping order** | `scaler.unscale_()` then `clip_grad_norm_` | Inside `train_model()` (no AMP) | Correct order for mixed precision |
| **Scheduler** | `CosineAnnealingLR(eta_min=lr*0.01)` | Inside `train_model()` | Explicit min LR floor |
| **Loss component tracking** | Per-component (signal, trend, conf, add) | Aggregate only | Better debugging |

## Estimated Speedup

On Colab T4 (16GB VRAM) training WaveFollower (~1.2M params, 4 TFs):

| Approach | Est. Time/Epoch | Bottleneck |
|---|---|---|
| WaveTrader_Colab style (batch=32, no AMP) | ~12 min | GPU idle, pandas overhead |
| WaveFollower style (batch=640, full AMP) | ~45 sec | GPU compute (ideal) |

**~16x faster per epoch** from the combined optimizations.

## Why Not Delegate to `train_model()` / `train_mtf_model()`?

The functions in `wavetrader/training.py` were written for the original FluxSignal/MTF models and:
1. Don't use mixed precision (`autocast` / `GradScaler`)
2. Don't call `scaler.unscale_()` before gradient clipping
3. Don't support `PreCachedMTFDataset` (assumes standard collation)
4. Don't track per-loss-component metrics
5. Don't use `non_blocking=True` for async transfers

The inline training loop in WaveFollower_Training.ipynb gives full control. All new models should follow this pattern.

## PreCachedMTFDataset — How It Works

The standard `MTFForexDataset` calls pandas operations in `__getitem__()` for every sample fetch. With batch_size=640 and num_workers=4, that's 2,560 pandas operations per batch — the CPU becomes the bottleneck.

`PreCachedMTFDataset` solves this by:
1. Iterating ALL samples once at startup (one-time cost)
2. Storing everything as pre-allocated contiguous `torch.Tensor` arrays
3. `__getitem__()` becomes a single tensor index operation — zero pandas

**RAM cost**: ~2-4 GB for 4 pairs × ~500k samples × 4 TFs

**Speed gain**: `__getitem__()` goes from ~5ms (pandas) to ~0.01ms (tensor index) — **500x faster per sample**.

## Checklist for New Training Notebooks

- [ ] Cell 1 has all 3 performance flags
- [ ] Model wrapped with `torch.compile(mode="reduce-overhead")`
- [ ] `PreCachedMTFDataset` used (not raw `MTFForexDataset`)
- [ ] `DataLoader`: `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=4`, `num_workers=4`
- [ ] Train loader: `drop_last=True`
- [ ] Val loader: `batch_size = 2 * train_batch_size`
- [ ] `AdamW(fused=True)` when on CUDA
- [ ] `GradScaler("cuda")` + `autocast("cuda")` wrapping forward pass
- [ ] `scaler.unscale_(optimizer)` BEFORE `clip_grad_norm_`
- [ ] `optimizer.zero_grad(set_to_none=True)` not `zero_grad()`
- [ ] `.to(device, non_blocking=True)` for batch tensors
- [ ] Inline training loop (not delegated to training.py functions)
- [ ] Batch size ≥ 128 (640 for T4, scale with GPU VRAM)
- [ ] LR scaled: `lr = base_lr * batch_size / 128`
