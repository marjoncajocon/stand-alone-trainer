#!/usr/bin/env python3
"""check_gpu_writer.py - verify train_gpu.py's .ntc reader and .ntw writer
against the Go implementation, WITHOUT needing a GPU or even torch.

This is the check README.md describes for the draughts path ("a Python-written
.ntw is byte-identical to a Go-written one"), extended to the chess kernel.
Only the PyTorch optimizer loop is untestable here; every byte-level contract
between Go and Python is.

What it does:
  1. Reads the .ntc with train_gpu.load_ntc and prints the counters, which must
     match what the Go loader printed when it packed the corpus.
  2. Reads the float weights out of a Go-written net.bin.
  3. Runs them through train_gpu.write_ntw.
  4. Requires the result to be BYTE-IDENTICAL to the Go-written .ntw.

Step 4 is the one that matters: it proves the two quantizers agree on every
scale, on the rounding rule (half away from zero, where numpy's default is
banker's), and on the file layout.

usage:
    python3 tools/check_gpu_writer.py --corpus data/chess.ntc \\
        --netbin nn_chess_h256.bin --ntw nn_chess_h256.ntw
"""

import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_gpu  # noqa: E402


def read_netbin(path):
    """chess-cli's net.bin: i32 hdr[4] then float w1, b1, w2, b2."""
    with open(path, "rb") as f:
        raw = f.read()
    magic, num_in, hid, ver = struct.unpack_from("<4i", raw, 0)
    if magic != 0x4E4E5545:
        raise SystemExit(f"{path}: bad net.bin magic {magic:#x}")
    p = 16
    n1 = num_in * hid
    w1 = np.frombuffer(raw, "<f4", count=n1, offset=p).astype(np.float64)
    p += n1 * 4
    b1 = np.frombuffer(raw, "<f4", count=hid, offset=p).astype(np.float64)
    p += hid * 4
    w2 = np.frombuffer(raw, "<f4", count=hid, offset=p).astype(np.float64)
    p += hid * 4
    (b2,) = struct.unpack_from("<f", raw, p)
    p += 4
    if p != len(raw):
        raise SystemExit(f"{path}: trailing {len(raw) - p} bytes")
    print(f"net.bin: {num_in}->{hid}->1  version={ver} "
          f"({'residue on eval_classical' if ver == 2 else 'full eval (legacy)'})")
    return num_in, hid, ver, w1.reshape(num_in, hid), b1, w2, b2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--netbin", required=True)
    ap.add_argument("--ntw", required=True)
    args = ap.parse_args()

    c = train_gpu.load_ntc(args.corpus)
    is_chess = c["kernel"] == train_gpu.KERNEL_CHESS768
    print(f"kernel: {'chess768' if is_chess else 'draughts'}")

    num_in, hid, ver, w1, b1, w2, b2 = read_netbin(args.netbin)
    if num_in != c["num_feat"]:
        raise SystemExit(f"net.bin has {num_in} inputs, corpus says {c['num_feat']}")
    if is_chess and ver != 2:
        raise SystemExit(f"net.bin version {ver}: expected 2 (residue). A v1 net "
                         f"has the whole classical eval baked in and must be retrained.")

    # write_ntw expects an embedding table with a trailing padding row, exactly
    # as train_gpu builds it from EmbeddingBag; it slices [:num_feat] back off.
    w1_pad = np.vstack([w1, np.zeros((1, hid))])
    w2_arr = w2.reshape(1, hid)

    tmp = args.ntw + ".pycheck"
    train_gpu.write_ntw(tmp, c["variant"], w1_pad, b1, w2_arr,
                        c["num_feat"], hid, c["buckets"],
                        kernel=c["kernel"], b2=b2)

    with open(tmp, "rb") as f:
        got = f.read()
    with open(args.ntw, "rb") as f:
        want = f.read()
    os.remove(tmp)

    if got == want:
        print(f"\nPASS: python-written .ntw is BYTE-IDENTICAL to the Go one "
              f"({len(got)} bytes)")
        return 0

    print(f"\nFAIL: python .ntw differs from Go's "
          f"(python {len(got)} bytes, go {len(want)} bytes)")
    for i in range(min(len(got), len(want))):
        if got[i] != want[i]:
            print(f"  first difference at byte {i}: python {got[i]:#04x} "
                  f"go {want[i]:#04x}")
            break
    return 1


if __name__ == "__main__":
    sys.exit(main())
