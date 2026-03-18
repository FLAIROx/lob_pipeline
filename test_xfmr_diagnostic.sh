#!/bin/bash
#SBATCH --job-name=xfmr-diag
#SBATCH --partition=workq
#SBATCH --account=brics.s5e
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=test_xfmr_diagnostic_%j.out
#SBATCH --error=test_xfmr_diagnostic_%j.err
#SBATCH --exclude=nid[010696-010718],nid010152,nid010110,nid[011112-011115],nid011294,nid[010083-010086],nid[010561-010564],nid010655,nid010052,nid010442,nid010851,nid010499,nid010463

set -euo pipefail
export TMPDIR="/tmp"
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

cd /lus/lfs1aip2/projects/s5e/lob_pipeline/LOBS5
/lus/lfs1aip2/projects/s5e/quant/miniforge3/envs/lobs5/bin/python -u - << 'PYEOF'
"""
Compare __call_ar__ vs __call_rnn__ — fixed version.

Uses the SAME data prep as training (repeat_book aligns book to message length)
and properly concatenates all RNN chunks.
"""
import sys, os
import jax, jax.numpy as jnp, numpy as np
from lob.encoding import Message_Tokenizer, Vocab
from lob.init_train import init_train_state, load_metadata, load_checkpoint
import lob.validation_helpers as valh

ckpt_path = "/lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_H4-self-attention/checkpoints/j2496072_ti7ek5z5_2496072"
data_dir = "/lus/lfs1aip2/projects/s5e/lob_pipeline/data/GOOG_jan2026"

v = Vocab()
n_classes = len(v)
book_dim = 503
n_cond_msgs = 20
n_eval_msgs = 20
msg_len = Message_Tokenizer.MSG_LEN
seq_len = (n_eval_msgs - 1) * msg_len

args = load_metadata(ckpt_path)
args.num_devices = 1; args.bsz = 1
print(f"d_model={args.d_model}, n_heads={getattr(args,'n_heads','?')}, n_layers={args.n_layers}")
print(f"merging={getattr(args,'merging','?')}, mode={getattr(args,'mode','?')}")

new_train_state, model_cls = init_train_state(
    args, n_classes=n_classes, seq_len=seq_len, book_dim=book_dim, book_seq_len=seq_len)

_step = 28495
try:
    ckpt = load_checkpoint(new_train_state, ckpt_path, step=_step, train=False, partial_restore=True)
except Exception as e:
    print(f"StandardRestore failed: {e}")
    import json, ast, tensorstore as ts, orbax.checkpoint as ocp
    from lob.init_train import deduplicate_trainstate
    _abs = os.path.abspath(ckpt_path)
    _mngr = ocp.CheckpointManager(_abs, item_names=('state','metadata'), options=ocp.CheckpointManagerOptions())
    _meta = _mngr.restore(_step, args=ocp.args.Composite(metadata=ocp.args.JsonRestore()))['metadata']
    _sd = os.path.join(_abs, str(_step), 'state')
    _mj = json.loads(open(os.path.join(_sd, '_METADATA')).read())
    _ocdbt = os.path.exists(os.path.join(_sd, 'ocdbt.base_path')) or os.path.exists(os.path.join(_sd, 'manifest.ocdbt'))
    _z3 = _mj.get('use_zarr3', False)
    _flat = {tuple(ast.literal_eval(k)): v for k, v in _mj['tree_metadata'].items()}
    _raw = {}
    for kp in _flat:
        if kp[0] != 'params': continue
        drv = 'zarr3' if _z3 else 'zarr'
        spec = {'driver': drv, 'kvstore': {'driver': 'ocdbt', 'base': f'file://{_sd}', 'path': '.'.join(kp)}} if _ocdbt else {'driver': drv, 'kvstore': {'driver': 'file', 'path': os.path.join(_sd, '.'.join(kp))}}
        _raw[kp[1:]] = np.asarray(ts.open(spec, open=True).result().read().result())
    _params = {}
    for kp, arr in _raw.items():
        d = _params
        for key in kp[:-1]: d = d.setdefault(key, {})
        d[kp[-1]] = arr
    _dedup = deduplicate_trainstate(new_train_state)
    ckpt = _meta; ckpt['model'] = _dedup.replace(params=_params)

state = ckpt['model']
import chex; chex.clear_trace_counter()
model = model_cls(training=False, step_rescale=1.0)

# Load data — same as training dataloader
import lob.inference_no_errcorr as inference
ds = inference.get_dataset(data_dir, n_cond_msgs, n_eval_msgs, test_split=0.1)
m_seq, _, b_seq_pv, msg_raw, book_init = ds[[0]]
m_seq = jnp.array(m_seq)[0]       # (seq_len+1,) = (n_eval*24 + 1 - 24 + 1,) ??
b_seq_pv = jnp.array(b_seq_pv)[0]

import preproc
transform_batch = jax.jit(jax.vmap(preproc.transform_L2_state_gpu, in_axes=(0, None, None)), static_argnums=(1,2))
b_seq = transform_batch(b_seq_pv[None], 500, 100)[0]
print(f"m_seq: {m_seq.shape}, b_seq: {b_seq.shape}")
print(f"msg_len={msg_len}, n_msgs_in_seq={m_seq.shape[0]//msg_len}")

# Prepare inputs EXACTLY as training does (train_step → repeat_book)
from lob.train_helpers import repeat_book

m_inp = m_seq[:-1]   # input tokens (shifted right)
m_tgt = m_seq[1:]    # target tokens
b_inp = b_seq[:-1]   # book states (one per message, but repeated to match token count)

print(f"m_inp: {m_inp.shape}, m_tgt: {m_tgt.shape}, b_inp: {b_inp.shape}")

# Add batch dim
batch_m = m_inp[None]          # (1, T)
batch_b = b_inp[None]          # (1, n_books, 503)
batch_inputs = repeat_book(batch_m, batch_b, True)
print(f"After repeat_book: msg={batch_inputs[0].shape}, book={batch_inputs[1].shape}")

batch_int = (jnp.ones((1, batch_m.shape[1])), jnp.ones((1, batch_m.shape[1])))

# ── TEST 1: __call_ar__ ──
print("\n=== __call_ar__ (full forward, same as training) ===")
ar_logits = model.apply({"params": state.params}, *batch_inputs, *batch_int, method="__call_ar__")
ar_logits = ar_logits[0]  # remove batch dim
print(f"AR logits: {ar_logits.shape}")

# CE on targets
T = min(ar_logits.shape[0], m_tgt.shape[0])
ar_ce = float(-ar_logits[jnp.arange(T), m_tgt[:T]].mean())
print(f"AR CE = {ar_ce:.4f}")

# ── TEST 2: __call_rnn__ (KV cache, chunk by chunk) ──
print("\n=== __call_rnn__ (KV cache scan) ===")
n_heads = getattr(args, 'n_heads', 16)
d_model = args.d_model
max_cache_len = m_inp.shape[0] + msg_len  # enough for all tokens
nh = n_heads
while nh > 1 and d_model % nh != 0: nh -= 1
tcfg = {'n_heads': nh, 'head_dim': d_model // nh, 'max_cache_len': max_cache_len, 'dtype': jnp.float32}
d_book = getattr(args, 'd_book', 503)
nh_b = n_heads
while nh_b > 1 and d_book % nh_b != 0: nh_b -= 1
tcfg_b = {'n_heads': nh_b, 'head_dim': d_book // nh_b, 'max_cache_len': max_cache_len, 'dtype': jnp.float32}

init_hidden = model.initialize_carry(
    1, hidden_size=0,
    n_message_layers=args.n_message_layers,
    n_book_pre_layers=args.n_book_pre_layers,
    n_book_post_layers=args.n_book_post_layers,
    n_fused_layers=args.n_layers,
    h_size_ema=args.ssm_size_base,
    is_transformer=True,
    transformer_config=tcfg, transformer_config_book=tcfg_b)

# Process one message (24 tokens) at a time, matching conditioning scan
hidden = init_hidden
all_rnn_logits = []
n_chunks = m_inp.shape[0] // msg_len

for chunk_i in range(n_chunks):
    start = chunk_i * msg_len
    end = start + msg_len
    m_chunk = m_inp[start:end]                     # (24,) tokens
    # Book: keep 2D (1, 503) so vmap produces correct shape
    bi = min(chunk_i, b_inp.shape[0] - 1)
    b_chunk = b_inp[bi:bi+1]                       # (1, 503)
    hidden, logits = valh.apply_model(hidden, m_chunk, b_chunk, state, model, False, True)
    # logits shape: (1, 24, 2112) — batch=1, tokens=24, vocab=2112
    all_rnn_logits.append(logits[0])  # remove batch dim -> (24, 2112)

rnn_logits = jnp.concatenate(all_rnn_logits, axis=0)  # (n_chunks*24, 2112)
print(f"RNN logits: {rnn_logits.shape}")

rnn_ce = float(-rnn_logits[jnp.arange(T), m_tgt[:T]].mean())
print(f"RNN CE = {rnn_ce:.4f}")

# ── COMPARE ──
print("\n=== COMPARISON ===")
min_len = min(ar_logits.shape[0], rnn_logits.shape[0], T)
ar_cmp = ar_logits[:min_len]
rnn_cmp = rnn_logits[:min_len]
diff = jnp.abs(ar_cmp - rnn_cmp)
print(f"Max logit diff:  {float(diff.max()):.4f}")
print(f"Mean logit diff: {float(diff.mean()):.4f}")

# Per-message breakdown
for msg_i in range(min(5, n_chunks)):
    s = msg_i * msg_len; e = s + msg_len
    if e > min_len: break
    d = jnp.abs(ar_cmp[s:e] - rnn_cmp[s:e])
    # Also show per-message CE
    ar_msg_ce = float(-ar_cmp[s:e][jnp.arange(msg_len), m_tgt[s:e]].mean())
    rnn_msg_ce = float(-rnn_cmp[s:e][jnp.arange(msg_len), m_tgt[s:e]].mean())
    print(f"  Msg {msg_i}: max_diff={float(d.max()):.4f} mean_diff={float(d.mean()):.4f} | AR_CE={ar_msg_ce:.4f} RNN_CE={rnn_msg_ce:.4f}")

ar_top1 = jnp.argmax(ar_cmp, axis=-1)
rnn_top1 = jnp.argmax(rnn_cmp, axis=-1)
agreement = float((ar_top1 == rnn_top1).mean())
print(f"\nTop-1 agreement: {agreement:.4f} ({int((ar_top1==rnn_top1).sum())}/{min_len})")
print(f"Overall CE:  AR={ar_ce:.4f}  RNN={rnn_ce:.4f}  delta={abs(ar_ce-rnn_ce):.4f}")

if float(diff.max()) > 1.0:
    print("\n*** DIVERGENCE: __call_rnn__ and __call_ar__ produce different logits ***")
    print("*** The transformer KV-cache inference has a bug ***")
elif ar_ce > 5.0:
    print("\n*** AR CE is unexpectedly high — check data alignment ***")
else:
    delta = abs(ar_ce - rnn_ce)
    if delta < 0.1:
        print(f"\n[OK] Match (CE delta={delta:.4f}) — issue is autoregressive error accumulation")
    else:
        print(f"\n[WARN] CE delta={delta:.4f} — possible numerical divergence in KV cache")
PYEOF
