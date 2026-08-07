#!/usr/bin/env python3
"""train_gpu.py - GPU NNUE training on Colab, via PyTorch.

Why this exists: compiling CUDA needs nvcc, which needs an NVIDIA GPU. PyTorch
ships precompiled kernels and is preinstalled on Colab, so this path has NO
build step at all - no nvcc, no cgo, no linker.

Scope is deliberately narrow. This script ONLY does the optimization. Everything
that has been validated stays in the Go binary:

    trainer --pack        corpus -> .ntc      (split frozen inside the file)
    train_gpu.py          .ntc   -> .ntw      (this file: GPU training)
    trainer --load .ntw --convert             -> nnue_weights.h
    parity.ps1                                -> 3-way bit-exactness gate

So the artifact that ships is still produced and checked by code that has been
proven against the reference trainer on 350 boards.

It needs NO variant knowledge. Board geometry, the material anchor and the phase
bucket are all baked into the .ntc by the Go packer, and the mirror permutation
is uniform across every variant:  mirror(plane*S + s) = (plane^1)*S + (S-1-s).

usage:
    python3 train_gpu.py --corpus data/filipino.ntc --h 256 --epochs 40 \
        --out nn_h256.ntw --log train.log
"""

import argparse
import math
import os
import struct
import sys
import time

import numpy as np

# ── .ntc reader ─────────────────────────────────────────────────────────────
# Layout written by internal/corpus/corpus.go Save(). Little-endian throughout.
#
#   "NTC1" u32:version [16]byte:variant u32:cells u32:numFeat u32:buckets
#   u64 x6: lines, unique, nTrain, nValid, tbSkip, evalPos
#   u32:holdout f64:lambda u64:poolLen
#   u16 pool[poolLen]
#   then (nTrain + nValid) packed 19-byte records:
#       i32 off, i32 end, i16 anchor, u8 bucket, f64 y
# Train samples come first, then valid.

NTC_HEADER = 104
NTC_REC = np.dtype([("off", "<i4"), ("end", "<i4"), ("anchor", "<i2"),
                    ("bucket", "u1"), ("y", "<f8")])  # 19 bytes, packed


def load_ntc(path):
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:4] != b"NTC1":
        raise SystemExit(f"{path}: not an NTC1 corpus "
                         f"(got magic {raw[:4]!r}) - produce it with: trainer --pack")
    (ver,) = struct.unpack_from("<I", raw, 4)
    if ver != 1:
        raise SystemExit(f"{path}: unsupported version {ver}")
    variant = raw[8:24].split(b"\0")[0].decode()
    cells, num_feat, buckets = struct.unpack_from("<III", raw, 24)
    lines, unique, n_train, n_valid, tb_skip, eval_pos = struct.unpack_from("<6Q", raw, 36)
    holdout, = struct.unpack_from("<I", raw, 84)
    lam, = struct.unpack_from("<d", raw, 88)
    pool_len, = struct.unpack_from("<Q", raw, 96)

    p = NTC_HEADER
    pool = np.frombuffer(raw, dtype="<u2", count=pool_len, offset=p)
    p += pool_len * 2
    n = n_train + n_valid
    recs = np.frombuffer(raw, dtype=NTC_REC, count=n, offset=p)
    p += n * NTC_REC.itemsize
    if p != len(raw):
        raise SystemExit(f"{path}: trailing {len(raw) - p} bytes (truncated or corrupt)")

    print(f"corpus {os.path.basename(path)}: variant={variant} cells={cells} "
          f"feats={num_feat} buckets={buckets}")
    print(f"loaded {lines} lines -> {unique} UNIQUE positions "
          f"({lines / max(1, unique):.1f}x dedup) -> {n_train} train / {n_valid} valid; "
          f"skipped {tb_skip} TB-territory lines; {eval_pos} positions had eval labels "
          f"(lambda={lam:.2f})")
    return dict(variant=variant, cells=cells, num_feat=num_feat, buckets=buckets,
                pool=pool, recs=recs, n_train=int(n_train), n_valid=int(n_valid))


def pad_features(pool, recs, num_feat):
    """Turn the CSR pool into a dense (N, maxLen) index matrix.

    Padding uses index num_feat, an extra embedding row pinned to zero, so a
    padded slot contributes nothing to the accumulator. EmbeddingBag's
    padding_idx also stops it collecting gradient.
    """
    lens = (recs["end"] - recs["off"]).astype(np.int64)
    max_len = int(lens.max())
    n = len(recs)
    out = np.full((n, max_len), num_feat, dtype=np.int32)
    # Vectorised scatter: build a flat position index for every active feature.
    starts = np.repeat(recs["off"].astype(np.int64), lens)
    within = np.arange(lens.sum(), dtype=np.int64) - np.repeat(
        np.concatenate(([0], np.cumsum(lens)[:-1])), lens)
    rows = np.repeat(np.arange(n, dtype=np.int64), lens)
    out[rows, within] = pool[starts + within].astype(np.int32)
    return out, max_len


def mirror_table(num_feat):
    """mirror(plane*S + s) = (plane^1)*S + (S-1-s); identical for all variants.

    One extra entry maps the padding row to itself so mirroring a padded slot
    stays padded.
    """
    s = num_feat // 4
    f = np.arange(num_feat, dtype=np.int64)
    m = (f // s ^ 1) * s + (s - 1 - f % s)
    return np.concatenate([m, [num_feat]]).astype(np.int64)


# ── .ntw writer ─────────────────────────────────────────────────────────────
# Layout read by internal/emit/emit.go LoadBin():
#   "NTW1" [16]byte:variant i32:H i32:numFeat i32:buckets
#   i16 W1[numFeat*H]  i32 B1[H]  i16 W2[buckets*H]

def q_round(x):
    """Round half AWAY FROM ZERO, matching Go's math.Round.

    numpy's np.round is banker's rounding and would disagree on exact .5 values,
    which is a silent one-ulp difference in the shipped weights.
    """
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))


def write_ntw(path, variant, w1, b1, w2, num_feat, h, buckets):
    # Quantization contract with the C engine (see internal/model/model.go):
    #   W1q = round(W1*256) int16 clipped to +/-32000
    #   B1q = round(B1*256) int32
    #   W2q = round(W2*256) int16 clipped to +/-32000
    w1q = np.clip(q_round(w1[:num_feat] * 256), -32000, 32000).astype("<i2")
    b1q = q_round(b1 * 256).astype("<i4")
    w2q = np.clip(q_round(w2 * 256), -32000, 32000).astype("<i2")
    name = variant.encode()[:16].ljust(16, b"\0")
    with open(path, "wb") as f:
        f.write(b"NTW1")
        f.write(name)
        f.write(struct.pack("<iii", h, num_feat, buckets))
        f.write(w1q.reshape(-1).tobytes())
        f.write(b1q.reshape(-1).tobytes())
        f.write(w2q.reshape(-1).tobytes())
    return w1q, b1q, w2q


# ── raw text -> .ntc, by shelling out to the Go packer ──────────────────────
#
# The loader is NOT reimplemented here. Dedup by board, holdout by whole game,
# the TB-territory skip and the lambda blend are all subtle and already
# validated in Go; a second implementation would drift silently and every MSE
# comparison would become meaningless. So --data just runs the Go binary.

def find_packer(explicit):
    import shutil
    here = os.path.dirname(os.path.abspath(__file__))
    # Platform order matters: picking trainer_linux_x64 on Windows gets you
    # "WinError 193: not a valid Win32 application".
    names = (["trainer_win_x64.exe", "trainer_linux_x64"] if os.name == "nt"
             else ["trainer_linux_x64", "trainer_win_x64.exe"])
    cands = []
    if explicit:
        cands.append(explicit)
    for d in (here, os.getcwd()):
        cands += [os.path.join(d, n) for n in names]
    for n in names:
        w = shutil.which(n)
        if w:
            cands.append(w)
    for c in cands:
        if os.path.isfile(c):
            return c
    raise SystemExit(
        "error: --data needs the Go packer and it was not found.\n"
        "  Looked in: " + ", ".join(dict.fromkeys(os.path.dirname(x) for x in cands)) + "\n"
        "  Upload trainer_linux_x64 next to this script, or pass --packer /path/to/it,\n"
        "  or pack at home and pass --corpus instead.")


def resolve_corpus(args):
    import subprocess
    if args.corpus and args.data:
        raise SystemExit("error: pass --corpus OR --data, not both")
    if args.corpus:
        return args.corpus
    if not args.data:
        raise SystemExit("error: need --corpus (packed .ntc) or --data (raw .txt glob)")

    packer = find_packer(args.packer)
    # Predictable by default: <variant>.ntc in the CURRENT directory. Deriving a
    # location from the glob is guesswork and lands the corpus somewhere the
    # user did not ask for. --keep-corpus overrides.
    out = os.path.abspath(args.keep_corpus or f"{args.variant}.ntc")
    if os.path.isfile(out) and not args.keep_corpus:
        print(f"reusing existing {out} (delete it to repack)")
        return out
    print(f"packing {args.data} -> {out}\n  using {packer}")

    # Copy the packer to local disk before running it, ALWAYS, on POSIX.
    #
    # Google Drive is mounted noexec, so execution is refused at the mount level
    # no matter what the permission bits say - and os.access(X_OK) still returns
    # True there, which makes "check then run" useless. Copying 2 MB is cheaper
    # than trying to detect the condition.
    import shutil, tempfile
    if os.name != "nt":
        local = os.path.join(tempfile.mkdtemp(prefix="ntpack_"), os.path.basename(packer))
        shutil.copy2(packer, local)
        os.chmod(local, 0o755)
        print(f"  (copied to {local}: Drive is mounted noexec)")
        packer = local

    try:
        # --h is irrelevant to packing, but older builds of the Go binary
        # resolved it before the pack branch and refused to run without it on a
        # machine with no engine repo. Passing it costs nothing and lets this
        # work with a packer that has not been updated.
        r = subprocess.run([packer, f"--variant={args.variant}", "--data", args.data,
                            "--h", str(args.h), "--pack", "--out-corpus", out])
    except OSError as e:
        raise SystemExit(
            f"error: could not run the packer ({e}).\n"
            f"  Copy it to local disk yourself and point at that:\n"
            f"    !cp {args.packer or 'trainer_linux_x64'} /content/ && chmod +x /content/trainer_linux_x64\n"
            f"    ... --packer /content/trainer_linux_x64")
    if r.returncode != 0 or not os.path.isfile(out):
        raise SystemExit(f"error: packing failed (exit {r.returncode})")
    return out


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="GPU NNUE training. Give it either a packed --corpus, or raw "
                    "--data text plus --variant and it will pack for you.")
    ap.add_argument("--corpus", default="", help="packed .ntc from: trainer --pack")
    ap.add_argument("--data", default="",
                    help="glob of raw selfplay .txt (QUOTE IT). Packs first, using the Go "
                         "binary - the loader is not reimplemented here on purpose")
    ap.add_argument("--variant", default="filipino",
                    help="only needed with --data; a .ntc already carries its variant")
    ap.add_argument("--packer", default="",
                    help="path to trainer_linux_x64 (default: look next to this script, "
                         "then the CWD, then PATH)")
    ap.add_argument("--keep-corpus", default="",
                    help="with --data: where to save the .ntc so later runs skip packing")
    ap.add_argument("--out", default="", help="output .ntw (default nn_<variant>_h<H>.ntw)")
    ap.add_argument("--h", type=int, default=256, help="hidden units")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--opt", default="adam", choices=["adam", "adagrad"])
    ap.add_argument("--lr-schedule", default="cosine", choices=["cosine", "const"])
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--k", type=float, default=1.0 / 256.0, help="sigmoid scale K")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda", help="cuda | cpu")
    ap.add_argument("--log", default="", help="tee stdout here (DO IT - the MSE matters)")
    args = ap.parse_args()

    if args.log:
        class _Tee:
            def __init__(self, *s): self.s = s
            def write(self, d):
                for x in self.s:
                    x.write(d); x.flush()
            def flush(self):
                for x in self.s:
                    x.flush()
        sys.stdout = _Tee(sys.__stdout__, open(args.log, "w"))

    # Resolve (and if needed build) the corpus BEFORE importing torch: packing
    # prints progress and a bad --data should fail immediately rather than after
    # a multi-second CUDA init.
    corpus_path = resolve_corpus(args)

    import torch
    import torch.nn as nn

    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        print("error: --device cuda but no GPU is visible.")
        print("  Colab: Runtime > Change runtime type > T4 GPU (this restarts the VM).")
        print("  Or pass --device cpu.")
        sys.exit(3)
    print(f"torch {torch.__version__}  device={dev}"
          + (f"  {torch.cuda.get_device_name(0)}" if dev == "cuda" else ""))

    c = load_ntc(corpus_path)
    num_feat, buckets, H = c["num_feat"], c["buckets"], args.h
    feats_np, max_len = pad_features(c["pool"], c["recs"], num_feat)
    print(f"feature matrix: {feats_np.shape} (max {max_len} pieces per position)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    mir = torch.from_numpy(mirror_table(num_feat)).to(dev)
    feats = torch.from_numpy(feats_np).to(dev)                       # (N, L) int32
    anchor = torch.from_numpy(c["recs"]["anchor"].astype(np.float32)).to(dev)
    bucket = torch.from_numpy(c["recs"]["bucket"].astype(np.int64)).to(dev)
    y = torch.from_numpy(c["recs"]["y"].astype(np.float32)).to(dev)
    n_train, n_valid = c["n_train"], c["n_valid"]

    # Model. W1 is ONE shared table evaluated twice (board and mirror) - that is
    # what makes f(b) = g(b) - g(mirror(b)) structurally antisymmetric, so the
    # property survives quantization exactly. Never add a second head.
    emb = nn.EmbeddingBag(num_feat + 1, H, mode="sum", padding_idx=num_feat).to(dev)
    with torch.no_grad():
        emb.weight.uniform_(-0.3, 0.3)
        emb.weight[num_feat].zero_()
    b1 = nn.Parameter(torch.full((H,), 0.05, device=dev))
    w2 = nn.Parameter(torch.empty(buckets, H, device=dev).uniform_(-0.5, 0.5))

    params = list(emb.parameters()) + [b1, w2]
    opt = (torch.optim.Adam(params, lr=args.lr) if args.opt == "adam"
           else torch.optim.Adagrad(params, lr=args.lr))

    K = args.k

    def forward(idx):
        f = feats[idx].long()
        acc1 = emb(f) + b1
        acc2 = emb(mir[f]) + b1
        a1 = acc1.clamp(0, 1)
        a2 = acc2.clamp(0, 1)
        o = ((a1 - a2) * w2[bucket[idx]]).sum(1)
        return torch.sigmoid(K * (anchor[idx] + 64.0 * o))

    @torch.no_grad()
    def valid_mse():
        tot, seen = 0.0, 0
        for s in range(n_train, n_train + n_valid, 16384):
            idx = torch.arange(s, min(s + 16384, n_train + n_valid), device=dev)
            d = y[idx] - forward(idx)
            tot += float((d * d).sum())
            seen += len(idx)
        return tot / max(1, seen)

    # Anchor-only baseline: what the frozen material term alone scores. Any net
    # that cannot beat this is worse than counting pieces.
    with torch.no_grad():
        vi = torch.arange(n_train, n_train + n_valid, device=dev)
        d = y[vi] - torch.sigmoid(K * anchor[vi])
        print(f"valid MSE, anchor-only (material): {float((d * d).mean()):.6f}")

    print(f"variant={c['variant']}  H={H}  feats={num_feat}  buckets={buckets}  "
          f"opt={args.opt}  epochs={args.epochs}  batch={args.batch}  lr={args.lr}  "
          f"seed={args.seed}")

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    steps = n_train // args.batch
    t_start = time.time()

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        if ep <= args.warmup_epochs and args.warmup_epochs > 0:
            lr = args.lr * ep / (args.warmup_epochs + 1)
        elif args.lr_schedule == "cosine":
            span = max(1, args.epochs - args.warmup_epochs)
            t = (ep - 1 - args.warmup_epochs) / span
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * t))
        else:
            lr = args.lr
        for gp in opt.param_groups:
            gp["lr"] = lr

        perm = torch.randperm(n_train, generator=g).to(dev)
        for st in range(steps):
            idx = perm[st * args.batch:(st + 1) * args.batch]
            p = forward(idx)
            d = y[idx] - p
            loss = (d * d).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():          # keep the padding row at exactly zero
                emb.weight[num_feat].zero_()

        print(f"epoch {ep}: valid MSE {valid_mse():.6f}  (lr {lr:.2e}, "
              f"{time.time() - t0:.1f}s)")

    w1 = emb.weight.detach().cpu().numpy().astype(np.float64)
    b1n = b1.detach().cpu().numpy().astype(np.float64)
    w2n = w2.detach().cpu().numpy().astype(np.float64)

    out = args.out or f"nn_{c['variant']}_h{H}.ntw"
    w1q, b1q, w2q = write_ntw(out, c["variant"], w1, b1n, w2n, num_feat, H, buckets)

    # int16 accumulator safety. search.c sums into int16 seeded by narrowing the
    # int32 NN_B1; that is only legal while the pre-clamp sum stays inside
    # +/-32767. An overflow silently changes the eval with no symptom except
    # worse play, so this is checked before the header is ever generated.
    worst_hi = int(b1q.max() + np.sort(w1q, axis=0)[-max_len:].sum(0).max())
    worst_lo = int(b1q.min() + np.sort(w1q, axis=0)[:max_len].sum(0).min())
    print(f"accumulator bound (worst case over {max_len} pieces): "
          f"HI {worst_hi:+d}  LO {worst_lo:+d}  (int16 limit +/-32767)")
    if worst_hi > 32767 or worst_lo < -32767:
        print("ACCUMULATOR OVERFLOW - do not install this net. Lower --lr or add clipping.")
        sys.exit(4)

    print(f"wrote {out}  ({time.time() - t_start:.0f}s total)")
    print()
    print("Next, on the machine that ships it:")
    print(f"  trainer --variant={c['variant']} --load {out} --convert --outh nnue_weights.h")
    print(f"  parity.ps1 -Bin {out} -Variant {c['variant']}      # MUST print PARITY: PASS")


if __name__ == "__main__":
    main()
