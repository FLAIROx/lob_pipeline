# OCDBT Checkpoint Loading Fix: 128-GPU → 1-GPU Inference

## Problem

LOBS5 checkpoints saved on multi-GPU runs (e.g. 128 GPUs / 32 processes) cannot be loaded for single-GPU inference. The checkpoint uses Orbax's pure OCDBT format (`manifest.ocdbt` + 32 process shards + `_METADATA` JSON) and does **not** contain the legacy `checkpoint` aggregate file that Orbax 0.11.33 expects.

**Affected checkpoint**: `vocal-sun-271_bkt2onyy` (step 20), saved by `lobs5` training pipeline.

**Error chain**:
1. `StandardRestore` → `FileNotFoundError: Checkpoint structure file does not exist at .../state.` (from `pytree_checkpoint_handler.py:1198` in `_read_aggregate_file`)
2. The existing fallback in `init_train.py` only caught `(ValueError, TypeError)`, so `FileNotFoundError` propagated uncaught.

## Root Cause

Orbax 0.11.33's `PyTreeCheckpointHandler._read_aggregate_file()` raises `FileNotFoundError` for OCDBT checkpoints that lack the legacy `checkpoint` msgpack file. This is a known limitation of older Orbax versions — newer versions handle OCDBT-only checkpoints natively.

### Why simply adding `FileNotFoundError` to the except isn't enough

The fix required bypassing Orbax's `restore()` entirely. Three monkey-patch approaches were tried and failed:

| Approach | Failure Mode |
|----------|-------------|
| Patch `_read_aggregate_file` → return `{}` | `_get_internal_metadata` does `flat_aggregate[tuple_key]` for each metadata key → `KeyError` on empty aggregate |
| Patch `_get_internal_metadata` + swap `ArrayHandler→NumpyHandler` in registry | `_maybe_deserialize` has tree structure mismatches between `flat_restored` (deserialized params only) and `structure` (full tree including `skip_deserialize=True` opt_state leaves with `aggregate_value=None`) → `KeyError` at `from_flat_dict` |
| Same as above but preserve `skip_deserialize` from metadata | Some opt_state metadata has `restore_type="None"` (inline aggregate values) → `Unknown type: "None"` in handler registry lookup |

## Solution: Direct TensorStore Read

**File**: `/lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/lob/init_train.py`

The fix adds `FileNotFoundError` to the outer except clause and bypasses Orbax's `restore()` entirely in the inference fallback. Instead, it:

1. Restores JSON metadata (config dict) via `CheckpointManager` — this always works
2. Gets abstract tree structure via `PyTreeCheckpointer.metadata()` — uses `_get_user_metadata()` which reads `_METADATA` JSON directly (no aggregate file needed)
3. Reads each `params.*` array directly via TensorStore with OCDBT driver — no sharding, no handler registry, no `_maybe_deserialize` complexity
4. Rebuilds nested params dict and grafts onto TrainState

### Key code path
```python
# 1. Read abstract tree (no aggregate needed)
abstract = ckptr.metadata(state_dir)
flat_abstract = _ocp_utils.to_flat_dict(abstract)

# 2. Read each param via TensorStore
for keypath, meta in flat_abstract.items():
    if keypath[0] != 'params':
        continue  # Skip opt_state for inference
    tspec = _th.get_tensorstore_spec(
        str(state_dir), name='.'.join(keypath),
        use_ocdbt=is_ocdbt, use_zarr3=use_zarr3,
    )
    t = ts.open(ts.Spec(tspec), open=True, context=ts_ctx).result()
    raw_params[keypath[1:]] = np.asarray(t.read().result())
```

### Why this works
- `metadata()` → `_get_user_metadata()` → `_read_metadata_file()` — reads `_METADATA` JSON directly, never touches `_read_aggregate_file`
- TensorStore OCDBT driver reads from `manifest.ocdbt` which indexes all 32 process shards transparently
- Reading as `np.ndarray` via TensorStore avoids JAX sharding requirements entirely
- Only reads 167 param arrays (skips ~300+ opt_state arrays) — faster than full restore

## Verification

- **Job 2432870**: Checkpoint loaded successfully, 167 param arrays read
- Log output:
  ```
  [load_checkpoint] StandardRestore failed (...), falling back to direct restore for inference
  [load_checkpoint] Reading 167 param arrays via TensorStore (OCDBT=True)
  [load_checkpoint] Loaded 167 param arrays
  [Rank 0/4] Processing 64 sequences out of 256 total
  ```
- Inference started and reached `sample_new()` — the checkpoint loading fix is complete

## Remaining Issue (Separate from checkpoint fix)

After checkpoint loads, inference hits a **pre-existing `TypeError`** in `preproc.py:94`:
```
TypeError: Indexer must have integer or boolean type, got indexer with type float32
```

This is in `transform_L2_state_gpu()`:
```python
mybook = mybook.at[book[:, 0]].set(book[:, 1])  # book[:, 0] is float32, needs int
```

This is a data preprocessing issue (input book data has float32 where int32 is expected), not related to the checkpoint loading fix. It likely existed before but was never reached because checkpoint loading always failed first.

## Files Changed

| File | Change |
|------|--------|
| `lob/init_train.py` (in LOBS5) | Added `FileNotFoundError` to outer except; replaced Orbax monkey-patch fallback with direct TensorStore param read |
| `pipeline/config.sh` | Updated stale `GOOG_DATA` path to correct `GOOGJAN2023_encoded` directory |
