# stand-alone-trainer

One NNUE trainer for **all six checkers/draughts engines** in this workspace.
Written in Go, cross-compiles from Windows with no C toolchain, and grows an
optional CUDA backend when built on Colab.

Everything is flat — binaries, scripts and `data/` all sit at the kit root.

```powershell
BUILD.bat                                   # builds trainer_win_x64.exe + trainer_linux_x64
.\trainer_win_x64.exe --list-variants
```

| Variant | Engine | Board | Feats | W2 buckets | Man/King |
|---|---|---|---|---|---|
| filipino | `dama-cli` | 8x8 diag | 128 | 4 | 100 / 320 |
| brazilian | `brazilian-cli` | 8x8 diag | 128 | 4 | 100 / 320 |
| english | `english-dama-cli` | 8x8 diag | 128 | **1** | 100 / **150** |
| russian | `rusian-dama-cli` | 8x8 diag | 128 | **1** | 100 / 320 |
| international | `international-dama-cli` | **10x10** | **200** | **1** | 100 / 320 |
| turkish | `turkish-cli` | 8x8 **all sq** | **256** | 4 | 100 / 320 |

Every constant was read out of that repo's own trainer, not guessed. **Buckets
is what that engine's `search.c` implements today** — `english`, `russian` and
`international` have no `NN_W2_BUCKETS` branch at all, so they get a single W2
and the generated header omits the `#define` entirely. Ask for more and the
trainer warns you the header will not compile there.

## Where the data goes

```
data\<variant>\*.txt        or a packed  data\<variant>.ntc
```

No flags needed — see **[DATA.md](DATA.md)**:

```powershell
.\trainer_win_x64.exe --variant=filipino --h 256
```

Same convention as `stand-alone-selfplay`'s `--out-dir`, so its output copies
straight across. A packed `.ntc` beats loose text when both are present.

## GPU training — use `train_gpu.py`

**This is the supported GPU path.** PyTorch ships precompiled CUDA kernels and is
preinstalled on Colab, so there is **no build step at all**: no nvcc, no cgo, no
linker. Python does only the optimization; everything validated stays in Go.

| Step | Tool |
|---|---|
| corpus -> `.ntc` | Go `trainer --pack` (freezes the split) |
| `.ntc` -> `.ntw` on GPU | **`train_gpu.py`** |
| `.ntw` -> `nnue_weights.h` | Go `trainer --load ... --convert` |
| parity gate | Go + C, `parity.ps1` |

On Colab (**Runtime > Change runtime type > T4 GPU** first — it restarts the VM):

```bash
python3 train_gpu.py --corpus corpus.ntc --h 256 --epochs 40 --out nn_h256.ntw --log train.log
```

Then, at home:

```powershell
.\trainer_win_x64.exe --variant=filipino --load nn_h256.ntw --convert --outh nnue_weights.h
.\parity.ps1 -Bin nn_h256.ntw -Variant filipino
```

`train_gpu.py` needs no variant knowledge: geometry, material anchor and phase
bucket are all baked into the `.ntc`, and the mirror permutation is uniform
across every variant.

**Verified locally** (numpy only, no GPU needed): `.ntc` reader counts match the
Go loader exactly; padded features match the CSR pool on 500 random rows; the
mirror table is an involution, a permutation, and identical to the Go formula;
`q_round` matches Go's half-away-from-zero `math.Round` where numpy's `np.round`
would not; and a Python-written `.ntw` is **byte-identical** to a Go-written one
and **passes the 350-board 3-way parity gate**. Only the PyTorch loop itself is
untested here.

### The Go `--gpu` flag is a dead end — do not use it

`internal/gpu/kernels.cu` and `build_colab.sh` are a hand-written CUDA backend
that was never made to link. It is kept for reference only. `--gpu=1` on any
binary built by `BUILD.bat` exits 3, which is correct. Use `train_gpu.py`.

The CPU binaries are genuinely self-contained: `trainer_linux_x64` is a static
ELF64 with no dynamic loader, so it runs anywhere including Colab.

## Two algorithms, and why both exist

```
--algo=online      AdaGrad, batch size 1, float64, single-threaded   [default]
--algo=minibatch   Adam/AdaGrad, batch 8192, float32, goroutine pool, GPU-capable
```

`online` is a near-verbatim transcription of the reference trainer's loop. It is
**not** the production path — it is the oracle. Because both it and the reference
are Go using the same `math/rand` and `math.Exp`, it can reproduce historical
nets, which is the only way to tell "the new optimizer is worse" apart from "the
rewrite has a bug".

`--gpu` requires `--algo=minibatch` and refuses otherwise, with the reason:
batch-1 training is one sequential update per position, each touching ~14 feature
rows. That is latency-bound and would run **slower** on a GPU than on a CPU.
Minibatching is what creates the parallelism; the GPU only harvests it.

**`--threads` is a pure performance knob.** Gradients are partitioned into a
fixed 16 chunks regardless of thread count and reduced in chunk order, so results
are byte-identical at any `--threads`. Verified: threads 1, 4 and 16 all produced
sha256 `1B975BED...`.

## The gate chain — nothing ships until all of it passes

```powershell
.\parity.ps1 -Bin nn_filipino_h256.ntw
```

runs the first three at once. In order of what they catch:

1. **Probe parity, int32.** Our integer reference vs C compiled against the
   header **we generate**. Validates feature extraction, the mirror permutation,
   the bucket function, quantization, and the header writer in one shot.
   *Status: 350/350 boards exact, on both H=128 and H=256 nets, and also against
   the original Go trainer.*
2. **Probe parity, int16.** The same boards through the engine's **narrow**
   accumulator. `search.c` sums into `int16` seeded by narrowing the int32
   `NN_B1`; that is only legal while the pre-clamp sum stays inside +/-32767, and
   an overflow silently changes the eval with no symptom except worse play.
   *Status: 350/350 exact.*
3. **Accumulator bound.** Printed by every training run, and fatal if exceeded.
   Still run `c_port/tools/probes/probe_accbound.exe` too — it scans a different
   position set.
4. **Holdout MSE of the QUANTIZED net.** The decisive metric, because it measures
   the artifact that actually plays. Always pass `--log`.
5. **Games.** Fixed-depth 10, ~2000 games vs the incumbent, then casual. MSE is
   necessary, not sufficient.

> `probe_nnue.exe` as shipped in `dama-cli` proves nothing about a new net: it
> `#include`s `nnue_weights.h` at **compile** time, so a prebuilt copy measures
> whatever header was next to it that day. `parity.ps1` always compiles a fresh
> probe against a fresh header. Same trap `note.md` records for `probe_accbound`.

## Status — what is verified and what is not

**Verified on this machine:**

- 3-way probe parity, 350 boards, two architectures, zero mismatches
- `--threads` determinism, byte-identical at 1 / 4 / 16
- `.ntc` pack round-trip exact: identical loader counters and identical epoch-1 MSE
- holdout split is location-independent: absolute glob and relative glob both
  give 327,837 train / 17,497 valid
- end-to-end training on the live filipino corpus; header emits, accbound clears
- both binaries build; `go vet` and `go test` clean

**NOT yet verified — do not treat as working:**

- **The CUDA path has never been compiled or run.** There is no nvcc on this
  machine. `internal/gpu/kernels.cu` is written to mirror
  `train.Batch.accumulate` exactly, but it is unexercised code. First Colab run
  should be the byte-identity check, not a training run: `--algo=minibatch
  --epochs=1 --seed=42` on CPU with `--threads 1` and `--threads 8`, and on GPU,
  then compare the three `.ntw` files. All three must match.
- **Minibatch hyperparameters are uncalibrated.** Nobody has shown minibatch
  matches `online` quality. Sweep `--batch` / `--lr` / `--epochs` on the same
  packed corpus and require a 3-seed MSE win before trusting it.
- **Only `filipino` has a live corpus.** The other five descriptors are correct
  by construction and by the parity gate, but untested against real data.

## MSE numbers are corpus-relative

The historical incumbents (H=32 `0.059633`, H=64 `0.059077`, H=128 `0.058732`)
were measured on the **old 145-file corpus, which no longer exists** — every file
in `dama-cli`'s data dir is now the 8/6 fixed-depth regeneration. On today's
corpus even anchor-only scores `0.050001`, and eval labels are present on ~99% of
lines rather than 3.7%, so targets are far more predictable. **A number from this
trainer is not comparable to `0.058732`.** Compare against a fresh reference run
on the same data — which is exactly what `--algo=online` is for.

## Layout

```
trainer_win_x64.exe / trainer_linux_x64 / trainer_colab   built binaries
build_all.ps1  BUILD.bat  UPDATE.bat    Windows + Linux builds
build_colab.sh                          the CUDA build (run on Colab)
train_nnue.ipynb                        Colab notebook
parity.ps1                              the acceptance gate
data/<variant>/                         corpora (gitignored)

cmd/trainer/          CLI
internal/variant/     the six descriptors - the ONLY place geometry differs
internal/corpus/      load, dedup, holdout, .ntc pack
internal/model/       weights, quantization, int32 + int16 references, accbound
internal/emit/        nnue_weights.h writer, .ntw, legacy DNN1/DNN2 readers
internal/train/       online.go (oracle), batch.go (production)
internal/gpu/         interface, !cuda stub, cuda cgo wrapper, kernels.cu
```

Adding a variant is one struct literal in `internal/variant/variant.go`.
