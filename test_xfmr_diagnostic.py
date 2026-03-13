#!/usr/bin/env python3
"""
Transformer __call_rnn__ vs __call__ Diagnostic
================================================
Tests 1-3: Isolated TransformerBlock (no checkpoint, no vmap)
Test 4:    Full model with actual checkpoint

Usage: python test_xfmr_diagnostic.py  (from LOBS5/ working dir)
"""
import os, sys, traceback, time as _time

REPO_DIR = "/lus/lfs1aip2/projects/s5e/lob_pipeline"
LOBS5_DIR = os.path.join(REPO_DIR, "LOBS5")
os.chdir(LOBS5_DIR)
if LOBS5_DIR not in sys.path:
    sys.path.insert(0, LOBS5_DIR)

# AlphaTrade submodule (needed by init_train imports)
for _cand in [os.path.join(LOBS5_DIR, 'AlphaTrade'),
              os.path.join(REPO_DIR, 'AlphaTrade'),
              os.path.join(os.path.dirname(REPO_DIR), 'AlphaTrade')]:
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.append(_cand)

import jax
import jax.numpy as jnp
import numpy as np

print(f"JAX {jax.__version__}  |  Devices: {jax.devices()}")
print(f"CWD: {os.getcwd()}")
print()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_PASS = 0
_FAIL = 0

def check(name, cond, detail=""):
    global _PASS, _FAIL
    tag = "PASS" if cond else "FAIL"
    if cond:
        _PASS += 1
    else:
        _FAIL += 1
    print(f"  [{tag}] {name}  {detail}")

def print_tree_shapes(tree, prefix="    "):
    """Recursively print shapes of all leaves in a pytree."""
    if isinstance(tree, (list, tuple)):
        kind = "list" if isinstance(tree, list) else "tuple"
        print(f"{prefix}({kind} len={len(tree)})")
        for i, item in enumerate(tree):
            print_tree_shapes(item, f"{prefix}  [{i}] ")
    elif hasattr(tree, 'shape'):
        print(f"{prefix}shape={tree.shape}  dtype={tree.dtype}")
    else:
        print(f"{prefix}{type(tree).__name__}: {tree}")


# ===================================================================
# TEST 1 — Single-layer __call__ vs __call_rnn__
# ===================================================================
def test1():
    print("=" * 70)
    print("TEST 1: TransformerBlock  __call__  vs  __call_rnn__")
    print("=" * 70)

    from s5.transformer import TransformerBlock

    H, nh, d_ff, hd = 512, 8, 2048, 64
    L = 24
    MAX_CACHE = 100

    for dtype_label, dtype, tol, tol_last, tol_tok in [
        ("float32", jnp.float32, 5e-4, 1e-5, 5e-3),
        ("bfloat16", jnp.bfloat16, 5e-2, 5e-2, 5e-2),
    ]:
        print(f"\n  --- dtype={dtype_label}  tol_all={tol}  tol_last={tol_last}  tol_tok={tol_tok} ---")

        block = TransformerBlock(H=H, n_heads=nh, d_ff=d_ff,
                                 training=False, dtype=dtype, use_flash=False)

        rng = jax.random.PRNGKey(42)
        x = jax.random.normal(rng, (L, H))
        variables = block.init(jax.random.PRNGKey(0), x)

        # Path A: __call__ (parallel, full causal mask)
        out_fwd = block.apply(variables, x)

        # Path B: __call_rnn__ (full chunk at pos=0)
        k0 = jnp.zeros((nh, MAX_CACHE, hd))
        v0 = jnp.zeros((nh, MAX_CACHE, hd))
        p0 = jnp.int32(0)
        _, out_rnn = block.apply(variables, (k0, v0, p0),
                                 x, jnp.zeros(L, dtype=bool),
                                 method="__call_rnn__")

        # Path C: __call_rnn__ (token-by-token)
        k1 = jnp.zeros((nh, MAX_CACHE, hd))
        v1 = jnp.zeros((nh, MAX_CACHE, hd))
        p1 = jnp.int32(0)
        tok_outs = []
        for t in range(L):
            (k1, v1, p1), ot = block.apply(
                variables, (k1, v1, p1),
                x[t:t+1], jnp.zeros(1, dtype=bool),
                method="__call_rnn__")
            tok_outs.append(ot[0])
        out_tok = jnp.stack(tok_outs)

        # Cast to f32 for comparison
        a = out_fwd.astype(jnp.float32)
        b = out_rnn.astype(jnp.float32)
        c = out_tok.astype(jnp.float32)

        d_ab_all  = float(jnp.max(jnp.abs(a - b)))
        d_ac_all  = float(jnp.max(jnp.abs(a - c)))
        d_bc_all  = float(jnp.max(jnp.abs(b - c)))
        d_ab_last = float(jnp.max(jnp.abs(a[-1] - b[-1])))
        d_ac_last = float(jnp.max(jnp.abs(a[-1] - c[-1])))

        check(f"fwd vs rnn_full  (all, {dtype_label})", d_ab_all < tol, f"max={d_ab_all:.2e}")
        check(f"fwd vs rnn_tok   (all, {dtype_label})", d_ac_all < tol_tok, f"max={d_ac_all:.2e}")
        check(f"rnn_full vs tok  (all, {dtype_label})", d_bc_all < tol_tok, f"max={d_bc_all:.2e}")
        check(f"fwd vs rnn_full  (last, {dtype_label})", d_ab_last < tol_last, f"max={d_ab_last:.2e}")
        check(f"fwd vs rnn_tok   (last, {dtype_label})", d_ac_last < tol_tok, f"max={d_ac_last:.2e}")


# ===================================================================
# TEST 2 — KV Cache Integrity
# ===================================================================
def test2():
    print("\n" + "=" * 70)
    print("TEST 2: KV Cache Integrity (2 × 24-token chunks)")
    print("=" * 70)

    from s5.transformer import TransformerBlock

    H, nh, d_ff, hd = 512, 8, 2048, 64
    MAX_CACHE = 200

    block = TransformerBlock(H=H, n_heads=nh, d_ff=d_ff,
                             training=False, dtype=jnp.float32, use_flash=False)

    x_full = jax.random.normal(jax.random.PRNGKey(99), (48, H))
    variables = block.init(jax.random.PRNGKey(1), x_full[:24])

    k0 = jnp.zeros((nh, MAX_CACHE, hd))
    v0 = jnp.zeros((nh, MAX_CACHE, hd))
    p0 = jnp.int32(0)
    resets = jnp.zeros(24, dtype=bool)

    # Chunk 1 (tokens 0-23)
    (k1, v1, p1), out1 = block.apply(
        variables, (k0, v0, p0), x_full[:24], resets, method="__call_rnn__")

    check("pos=24 after chunk 1", int(p1) == 24, f"pos={int(p1)}")
    check("k_cache[:24] non-zero", bool(jnp.any(k1[:, :24, :] != 0)))
    check("k_cache[24:] all-zero", bool(jnp.all(k1[:, 24:, :] == 0)))

    # Chunk 2 (tokens 24-47)
    (k2, v2, p2), out2 = block.apply(
        variables, (k1, v1, p1), x_full[24:48], resets, method="__call_rnn__")

    check("pos=48 after chunk 2", int(p2) == 48, f"pos={int(p2)}")
    check("k_cache[:48] non-zero", bool(jnp.any(k2[:, :48, :] != 0)))
    check("k_cache[48:] all-zero", bool(jnp.all(k2[:, 48:, :] == 0)))

    # Chunk 1 entries must be preserved after chunk 2
    k_pres = float(jnp.max(jnp.abs(k1[:, :24, :] - k2[:, :24, :])))
    check("chunk-1 K preserved", k_pres == 0.0, f"max_diff={k_pres:.2e}")

    # Full __call__ at 48 tokens vs chunked __call_rnn__
    out_full = block.apply(variables, x_full)
    d_last = float(jnp.max(jnp.abs(out_full[-1] - out2[-1])))
    check("full_fwd[-1] vs chunked[-1]", d_last < 1e-3, f"max_diff={d_last:.2e}")


# ===================================================================
# TEST 3 — MHDA Weight Access Verification
# ===================================================================
def test3():
    print("\n" + "=" * 70)
    print("TEST 3: MHDA Weight Access Verification")
    print("=" * 70)

    from s5.transformer import TransformerBlock, sinusoidal_positional_encoding

    H, nh, d_ff, hd = 512, 8, 2048, 64

    block = TransformerBlock(H=H, n_heads=nh, d_ff=d_ff,
                             training=False, dtype=jnp.float32, use_flash=False)

    x = jax.random.normal(jax.random.PRNGKey(7), (4, H))
    variables = block.init(jax.random.PRNGKey(2), x)

    attn_p = variables["params"]["attn"]

    # --- Shape checks ---
    check("Q kernel (512,8,64)", attn_p["query"]["kernel"].shape == (H, nh, hd),
          f"got {attn_p['query']['kernel'].shape}")
    check("K kernel (512,8,64)", attn_p["key"]["kernel"].shape == (H, nh, hd),
          f"got {attn_p['key']['kernel'].shape}")
    check("V kernel (512,8,64)", attn_p["value"]["kernel"].shape == (H, nh, hd),
          f"got {attn_p['value']['kernel'].shape}")
    check("Out kernel (8,64,512)", attn_p["out"]["kernel"].shape == (nh, hd, H),
          f"got {attn_p['out']['kernel'].shape}")

    # --- Projection equivalence ---
    h = x
    q_ein = jnp.einsum("...d,dnk->...nk", h, attn_p["query"]["kernel"])
    q_mm  = jnp.dot(h, attn_p["query"]["kernel"].reshape(H, -1)).reshape(-1, nh, hd)
    check("Q einsum == matmul", float(jnp.max(jnp.abs(q_ein - q_mm))) < 1e-6)

    dummy = jax.random.normal(jax.random.PRNGKey(13), (4, nh, hd))
    o_ein = jnp.einsum("...nk,nkd->...d", dummy, attn_p["out"]["kernel"])
    o_mm  = jnp.dot(dummy.reshape(4, -1), attn_p["out"]["kernel"].reshape(-1, H))
    check("Out einsum == matmul", float(jnp.max(jnp.abs(o_ein - o_mm))) < 1e-5)

    # --- Full manual replication vs __call__ ---
    # Use Flax LayerNorm (not jax.nn.standardize) for exact match
    from flax import linen as nn_flax
    L = x.shape[0]
    pe = sinusoidal_positional_encoding(L, H)
    x_pe = x + pe

    h_n = nn_flax.LayerNorm().apply({"params": variables["params"]["norm1"]}, x_pe)

    q = jnp.einsum("ld,dnk->lnk", h_n, attn_p["query"]["kernel"])
    k = jnp.einsum("ld,dnk->lnk", h_n, attn_p["key"]["kernel"])
    v = jnp.einsum("ld,dnk->lnk", h_n, attn_p["value"]["kernel"])
    if "bias" in attn_p.get("query", {}):
        q += attn_p["query"]["bias"]; k += attn_p["key"]["bias"]; v += attn_p["value"]["bias"]

    qt = jnp.transpose(q, (1, 0, 2))
    kt = jnp.transpose(k, (1, 0, 2))
    vt = jnp.transpose(v, (1, 0, 2))

    scale = jnp.sqrt(jnp.float32(hd))
    w = jnp.einsum("hqd,hkd->hqk", qt, kt) / scale
    mask = jnp.tril(jnp.ones((L, L), dtype=bool))
    w = jnp.where(mask[None], w, jnp.finfo(jnp.float32).min)
    w = jax.nn.softmax(w, axis=-1)
    a_out = jnp.transpose(jnp.einsum("hqk,hkd->hqd", w, vt), (1, 0, 2))

    out_proj = jnp.einsum("lnk,nkd->ld", a_out, attn_p["out"]["kernel"])
    if "bias" in attn_p.get("out", {}):
        out_proj += attn_p["out"]["bias"]
    x_mid = x_pe + out_proj

    h2 = nn_flax.LayerNorm().apply({"params": variables["params"]["norm2"]}, x_mid)

    ff_p = variables["params"]["ff"]
    h2 = jnp.dot(h2, ff_p["layers_0"]["kernel"]) + ff_p["layers_0"]["bias"]
    h2 = jax.nn.gelu(h2)
    h2 = jnp.dot(h2, ff_p["layers_2"]["kernel"]) + ff_p["layers_2"]["bias"]
    out_manual = x_mid + h2

    out_call = block.apply(variables, x)
    d_man = float(jnp.max(jnp.abs(out_call - out_manual)))
    check("full manual replication vs __call__", d_man < 1e-5, f"max_diff={d_man:.2e}")


# ===================================================================
# TEST 4 — Full Model with Checkpoint
# ===================================================================
def test4():
    print("\n" + "=" * 70)
    print("TEST 4: Full Model Logit Comparison  (with checkpoint)")
    print("=" * 70)

    CKPT = ("/lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/"
            "exp_H4-self-attention/checkpoints/j2496072_ti7ek5z5_2496072")
    STEP = 28495

    from lob.init_train import init_train_state, load_checkpoint, load_metadata
    from lob.encoding import Vocab, Message_Tokenizer
    from lob.lob_seq_model import PaddedLobPredModel
    from s5.transformer import TransformerBlock

    v = Vocab()
    n_classes = len(v)
    book_dim = 503
    msg_len = Message_Tokenizer.MSG_LEN
    n_msgs = 5
    seq_len = n_msgs * msg_len   # 120 tokens

    print(f"  Checkpoint: {CKPT}")
    print(f"  seq_len={seq_len} ({n_msgs} msgs x {msg_len} toks)")

    # --- Load metadata & model ---
    # load_metadata expects the checkpoint ROOT (not the step subdir)
    args = load_metadata(CKPT)
    args.num_devices = 1
    args.bsz = 1

    d_model  = args.d_model
    n_heads  = getattr(args, "n_heads", 16)
    d_ff_cfg = getattr(args, "d_ff", 0)
    if d_ff_cfg <= 0:
        d_ff_cfg = 4 * d_model

    print(f"  d_model={d_model}  n_heads={n_heads}  d_ff={d_ff_cfg}")
    print(f"  mode={args.mode}  dtype={getattr(args, 'dtype', 'float32')}")
    print(f"  layers: msg={args.n_message_layers}  book_pre={args.n_book_pre_layers}"
          f"  book_post={args.n_book_post_layers}  fused={args.n_layers}")

    state, model_cls = init_train_state(
        args, n_classes=n_classes, seq_len=seq_len,
        book_dim=book_dim, book_seq_len=seq_len)

    # Try loading checkpoint; fall back to random-init params (still valid
    # for comparing __call_ar__ vs __call_rnn__ code paths).
    try:
        ckpt = load_checkpoint(state, CKPT, step=STEP, train=False, partial_restore=True)
        state = ckpt["model"]
        print("  Checkpoint loaded OK")
    except Exception as ckpt_err:
        print(f"  Checkpoint load failed ({ckpt_err}), using random-init params")
        from lob.init_train import deduplicate_trainstate
        state = deduplicate_trainstate(state)

    model = model_cls(training=False, step_rescale=1.0)

    # Compute effective heads
    nh = n_heads
    while nh > 1 and d_model % nh != 0:
        nh -= 1
    hd = d_model // nh
    max_cache_len = seq_len + msg_len

    d_book = getattr(args, "d_book", 503)
    nh_book = n_heads
    while nh_book > 1 and d_book % nh_book != 0:
        nh_book -= 1
    hd_book = d_book // nh_book

    batchnorm = getattr(args, "batchnorm", False)
    apply_vars = {"params": state.params}
    if batchnorm:
        apply_vars["batch_stats"] = state.batch_stats

    # --- Random input ---
    rng = jax.random.PRNGKey(123)
    x_m = jax.random.randint(rng, (1, seq_len), 0, n_classes)
    x_b = jax.random.normal(jax.random.PRNGKey(456), (1, seq_len, book_dim))
    ts  = (jnp.ones((1, seq_len)), jnp.ones((1, seq_len)))

    # ---- PATH A: __call_ar__ (parallel, full causal mask) ----
    print("\n  [A] __call_ar__ ...")
    t0 = _time.time()
    try:
        logits_ar = model.apply(apply_vars, x_m, x_b, *ts, method="__call_ar__")
        print(f"      shape={logits_ar.shape}  ({_time.time()-t0:.1f}s)")
        ar_ok = True
    except Exception as e:
        print(f"      CRASHED: {e}")
        traceback.print_exc()
        ar_ok = False

    # ---- PATH B: __call_rnn__  (sample_new init — shows shape bug) ----
    print("\n  [B] __call_rnn__  (sample_new initialization) ...")

    transformer_config = {"n_heads": nh, "head_dim": hd,
                          "max_cache_len": max_cache_len, "dtype": jnp.float32}
    transformer_config_book = {"n_heads": nh_book, "head_dim": hd_book,
                               "max_cache_len": max_cache_len, "dtype": jnp.float32}

    init_hidden = PaddedLobPredModel.initialize_carry(
        batch_size=1, hidden_size=0,
        n_message_layers=args.n_message_layers,
        n_book_pre_layers=args.n_book_pre_layers,
        n_book_post_layers=args.n_book_post_layers,
        n_fused_layers=args.n_layers,
        h_size_ema=getattr(args, "ssm_size_base", d_model),
        is_transformer=True,
        transformer_config=transformer_config,
        transformer_config_book=transformer_config_book)

    print("      Shapes BEFORE resize:")
    print_tree_shapes(init_hidden)

    batch_size = 1
    init_hidden_batched = jax.tree_util.tree_map(
        lambda x: jnp.resize(x, (batch_size,) + x.shape), init_hidden)

    print("      Shapes AFTER resize (batch_size=1):")
    print_tree_shapes(init_hidden_batched)

    # Inspect a single k_cache leaf
    # Navigate: init_hidden_batched[0] = msg_hidden (list), [0] = first layer, [0] = k_cache
    sample_k = init_hidden_batched[0][0][0]
    print(f"\n      Sample k_cache leaf: shape={sample_k.shape}  ndim={sample_k.ndim}")
    print(f"      __call_rnn__ expects: (nh={nh}, max_cache={max_cache_len}, hd={hd})  ndim=3")
    print(f"      After vmap strips axis-0: shape will be {sample_k.shape[1:]}")
    expected_rnn_shape = (nh, max_cache_len, hd)
    actual_after_vmap = sample_k.shape[1:]
    shape_ok = actual_after_vmap == expected_rnn_shape
    check("cache shape matches __call_rnn__", shape_ok,
          f"after_vmap={actual_after_vmap} vs expected={expected_rnn_shape}")

    dones = (jnp.zeros((1, seq_len), dtype=bool),) * 3
    t0 = _time.time()
    try:
        rnn_h_b, logits_rnn_b = model.apply(
            apply_vars, init_hidden_batched,
            x_m, x_b, *dones, *ts, method="__call_rnn__")
        print(f"      shape={logits_rnn_b.shape}  ({_time.time()-t0:.1f}s)")
        rnn_b_ok = True
    except Exception as e:
        print(f"      CRASHED: {e}")
        traceback.print_exc()
        rnn_b_ok = False

    # ---- PATH C: __call_rnn__  (FIXED init — no leading-1 dims) ----
    print("\n  [C] __call_rnn__  (fixed initialization) ...")

    def _make_cache(cfg, bsz):
        """Cache without the extra leading-1 dim. After vmap → (nh, cache, hd)."""
        return (jnp.zeros((bsz, cfg["n_heads"], cfg["max_cache_len"], cfg["head_dim"]),
                          dtype=cfg.get("dtype", jnp.float32)),
                jnp.zeros((bsz, cfg["n_heads"], cfg["max_cache_len"], cfg["head_dim"]),
                          dtype=cfg.get("dtype", jnp.float32)),
                jnp.zeros((bsz,), dtype=jnp.int32))

    msg_h   = [_make_cache(transformer_config, batch_size)
               for _ in range(args.n_message_layers)]
    bk_pre  = [_make_cache(transformer_config_book, batch_size)
               for _ in range(args.n_book_pre_layers)]
    bk_post = [_make_cache(transformer_config, batch_size)
               for _ in range(args.n_book_post_layers)]
    fused_h = [_make_cache(transformer_config, batch_size)
               for _ in range(args.n_layers)]

    ema_sz  = getattr(args, "ssm_size_base", d_model)
    ema_st  = (jnp.zeros((batch_size, 1, ema_sz)),
               jnp.ones((batch_size, 1, 1)))

    fixed_hidden = (msg_h, (bk_pre, bk_post), fused_h, ema_st)

    print("      Fixed shapes:")
    print_tree_shapes(fixed_hidden)

    t0 = _time.time()
    try:
        rnn_h_c, logits_rnn_c = model.apply(
            apply_vars, fixed_hidden,
            x_m, x_b, *dones, *ts, method="__call_rnn__")
        print(f"      shape={logits_rnn_c.shape}  ({_time.time()-t0:.1f}s)")
        rnn_c_ok = True
    except Exception as e:
        print(f"      CRASHED: {e}")
        traceback.print_exc()
        rnn_c_ok = False

    # ---- Compare logits ----
    dtype_str = getattr(args, "dtype", "float32")
    tol = 0.5 if dtype_str == "bfloat16" else 1e-3

    if ar_ok and rnn_b_ok:
        a = logits_ar.astype(jnp.float32)
        b = logits_rnn_b.astype(jnp.float32)
        d_last = float(jnp.max(jnp.abs(a[0, -1] - b[0, -1])))
        d_all  = float(jnp.max(jnp.abs(a - b)))
        print(f"\n  __call_ar__  vs  __call_rnn__ (sample_new init):")
        print(f"      last-tok max diff = {d_last:.4e}")
        print(f"      all-tok  max diff = {d_all:.4e}")
        check("ar vs rnn_sampleNew (last)", d_last < tol, f"diff={d_last:.2e} tol={tol}")
        check("ar vs rnn_sampleNew (all)",  d_all  < tol, f"diff={d_all:.2e} tol={tol}")

    if ar_ok and rnn_c_ok:
        a = logits_ar.astype(jnp.float32)
        c = logits_rnn_c.astype(jnp.float32)
        d_last = float(jnp.max(jnp.abs(a[0, -1] - c[0, -1])))
        d_all  = float(jnp.max(jnp.abs(a - c)))
        print(f"\n  __call_ar__  vs  __call_rnn__ (fixed init):")
        print(f"      last-tok max diff = {d_last:.4e}")
        print(f"      all-tok  max diff = {d_all:.4e}")
        check("ar vs rnn_fixed (last)", d_last < tol, f"diff={d_last:.2e} tol={tol}")
        check("ar vs rnn_fixed (all)",  d_all  < tol, f"diff={d_all:.2e} tol={tol}")

    if rnn_b_ok and rnn_c_ok:
        b = logits_rnn_b.astype(jnp.float32)
        c = logits_rnn_c.astype(jnp.float32)
        d = float(jnp.max(jnp.abs(b - c)))
        print(f"\n  __call_rnn__ sample_new vs fixed:  max_diff={d:.4e}")
        check("sampleNew vs fixed", d < tol, f"diff={d:.2e}")

    # ---- Inspect hidden state pos values ----
    for label, h_state, ok in [("sample_new", rnn_h_b if rnn_b_ok else None, rnn_b_ok),
                                ("fixed",      rnn_h_c if rnn_c_ok else None, rnn_c_ok)]:
        if not ok:
            continue
        print(f"\n  Hidden-state pos values  ({label} init):")
        m_h, bk_h, fu_h, ema_h = h_state
        for i, layer_h in enumerate(m_h):
            _k, _v, _p = layer_h
            print(f"      msg[{i}]   pos={int(jnp.squeeze(_p)):>6d}  "
                  f"k_nonzero={bool(jnp.any(_k != 0))}")
        bk_pre_h, bk_post_h = bk_h
        for i, layer_h in enumerate(bk_pre_h):
            _k, _v, _p = layer_h
            print(f"      bk_pre[{i}] pos={int(jnp.squeeze(_p)):>6d}  "
                  f"k_nonzero={bool(jnp.any(_k != 0))}")
        for i, layer_h in enumerate(bk_post_h):
            _k, _v, _p = layer_h
            print(f"      bk_post[{i}] pos={int(jnp.squeeze(_p)):>6d}  "
                  f"k_nonzero={bool(jnp.any(_k != 0))}")
        for i, layer_h in enumerate(fu_h):
            _k, _v, _p = layer_h
            print(f"      fused[{i}]  pos={int(jnp.squeeze(_p)):>6d}  "
                  f"k_nonzero={bool(jnp.any(_k != 0))}")


# ===================================================================
# MAIN
# ===================================================================
if __name__ == "__main__":
    for test_fn in [test1, test2, test3, test4]:
        try:
            test_fn()
        except Exception as e:
            print(f"\n  !!! {test_fn.__name__} CRASHED: {e}")
            traceback.print_exc()
        print()

    print("=" * 70)
    print(f"SUMMARY:  {_PASS} passed    {_FAIL} failed")
    print("=" * 70)
