# stand-alone-trainer

One NNUE trainer for **all seven engines** in this workspace — the six
checkers/draughts variants and chess. Written in Go, cross-compiles from
Windows with no C toolchain, and grows an optional CUDA backend when built on
Colab.

Everything is flat — binaries, scripts and `data/` all sit at the kit root.

```powershell
BUILD.bat                                   # builds trainer_win_x64.exe + trainer_linux_x64
.\trainer_win_x64.exe --list-variants
```

| Variant | Engine | Board | Feats | W2 buckets | Anchor |
|---|---|---|---|---|---|
| filipino | `dama-cli` | 8x8 diag | 128 | 4 | man 100 / king 320 |
| brazilian | `brazilian-cli` | 8x8 diag | 128 | 4 | man 100 / king 320 |
| english | `english-dama-cli` | 8x8 diag | 128 | **1** | man 100 / king **150** |
| russian | `rusian-dama-cli` | 8x8 diag | 128 | **1** | man 100 / king 320 |
| international | `international-dama-cli` | **10x10** | **200** | **1** | man 100 / king 320 |
| turkish | `turkish-cli` | 8x8 **all sq** | **256** | 4 | man 100 / king 320 |
| **chess** | `chess-cli` | FEN | **768** | **1** | **`eval_classical`** |

Every constant was read out of that repo's own trainer, not guessed. **Buckets
is what that engine's `search.c` implements today** — `english`, `russian` and
`international` have no `NN_W2_BUCKETS` branch at all, so they get a single W2
and the generated header omits the `#define` entirely. Ask for more and the
trainer warns you the header will not compile there.

## Chess is the other KERNEL, not a seventh variant

The six draughts engines share one integer kernel. Chess does not, so it is a
second kernel family rather than another descriptor:

| | draughts | chess |
|---|---|---|
| input | board string, 4 absolute planes | FEN, 768 **stm-relative** features |
| symmetry | `f(b) = g(b) − g(mirror(b))`, at the OUTPUT | in the input encoding |
| accumulators | two (board + mirror) | **one** |
| output bias | none | **`b2`** |
| quantization | ×256, clamp `[0,256]` | **QA=255, QB=64**, clamp `[0,255]` |
| header symbols | `NN_W1/NN_B1/NN_W2` + `#define`s | `nnue_w1/b1/w2/b2`, flat, no defines |
| anchor | frozen material count | **`eval_classical()`**, a texel-tuned HCE |
| `--h` | reads the engine's shipped `NN_H` | **pinned 256** (`nnue.c` static-asserts it) |
| int16 accumulator gate | applies | **N/A** — that engine uses int32 too |

The anchor is the part that costs something. Chess's net is a *residue* on
`eval_classical()`, so the trainer has to compute that per position; the port
lives in `internal/chess/` and is gated exactly against `chess-cli/c_port` by
`anchor_parity.ps1`. **Re-run that gate after any texel retune** — the anchor is
frozen into the `.ntc` at pack time, so a drifted copy trains cleanly on Colab
and yields a net fitted against an eval the engine does not compute. There is no
other symptom.

Standard chess and Chess960 **pool into one net**, which is what `chess-cli`'s
own kit does: the eval is position-agnostic and the app ships a single net.
`--variant=chess` reads both `data\chess\` and `data\chess960\`.

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
.\parity.ps1 -Bin nn_filipino_h256.ntw        # the six draughts engines
.\anchor_parity.ps1                            # chess: the anchor. Before EVERY --pack.
.\parity_chess.ps1 -Bin nn_chess_h256.ntw      # chess: anchor + kernel + quantizer
```

`parity_chess.ps1` has three legs, all compiled fresh against the live
`chess-cli/c_port` sources: Go `eval_classical` vs `eval.c`; Go `ScoreCP` vs
`nnue.c` compiled against the header we generate; and our `net.bin` fed through
`chess-cli`'s **own `n2h.exe`**, whose header must come out byte-identical to
ours. That last one validates every quantization scale at once against the tool
that produced every shipped chess header so far — it is what makes `trainer.c`
and `n2h.c` genuinely redundant rather than merely duplicated.

It reports the int16 narrow-accumulator leg as **N/A** rather than passing it,
because chess's engine accumulates in int32 by design.

The draughts chain, in order of what it catches:

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

**Verified for chess specifically:**

- **anchor parity, 20,000 real corpus FENs, 0 mismatches** — Go `eval_classical`
  against a probe compiled from the live `chess-cli/c_port/eval.c`, strided
  across all 12 corpus files so the sample reaches endgames, not just openings
- **kernel parity, 400 FENs, 0 mismatches** — Go `ScoreCP` against `nnue.c`
  compiled against our generated header
- **quantizer: header body byte-identical to `n2h.exe`** (730,720 bytes)
- 50,000 FENs round-trip `FromFEN → ToFEN` byte-identically
- `--threads` determinism holds for the new kernel: 1 / 4 / 16 all produced the
  same `.ntw`
- a Python-written chess `.ntw` is **byte-identical** to the Go one
  (`tools/check_gpu_writer.py`), so `train_gpu.py`'s NTC2 reader, NTW2 writer and
  QA/QB scales all agree with Go
- end-to-end: pack → train → quantize → header, on a real 20k-line corpus slice;
  holdout MSE falls monotonically and beats the anchor-only baseline
- the six draughts variants are unaffected: `parity.ps1` still 350/350

**NOT yet verified — do not treat as working:**

- **No chess net has been trained on the full corpus, and none has played a
  game.** The gates above prove the artifact is *correct*, not that it is
  *stronger*. The decision to ship is still `abmatch.exe` vs the tuned classical
  engine on both standard and 960 — reject <48%, accept ≥51.5%.
- **The shipped app is classical-only today.** `chess_master`'s Android
  `CMakeLists.txt` compiles `eval.c` but not `nnue.c` and defines no
  `USE_NNUE`. Turning that on is a separate decision, out of scope here.
- **The PyTorch loop for chess has not been run**, only its byte-level contracts
  with Go. Same status as the draughts path.

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

## A bug this replaces: chess-cli's trainer.c drops ~19% of every corpus

`chess-cli/c_port/tools/trainer.c:106` finds the result token by substring
search, with the comment *"FENs never contain these substrings"*. That is
false. A FEN's placement field contains **`1/2`** whenever a rank ends with one
empty square and the next begins with two — `rnbqkbn1/2pppppp/...`. `strstr`
then matches inside the FEN, line 126 truncates the string there, `pos_from_fen`
fails, and the line is **silently discarded**.

Measured on 8 corpus files: 2,245,788 lines total. This trainer reads all of
them; `trainer.c` reads 1,825,852 and drops **419,936 — exactly the number of
lines whose placement field contains `1/2`**, matching to the line.

It is not a random 19% either. It selects for a specific emptiness pattern, so
the discarded set is structurally biased rather than a uniform sample. Every
chess NNUE net trained with that kit was fitted on the surviving 81%.

This loader parses by FIELD POSITION (`fenSpan`), so it cannot go wrong the same
way — and it is why the "line" counters here will not match a historical
`trainer.c` log on the same data.

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
internal/variant/     the seven descriptors + the Kernel discriminator
internal/corpus/      load, dedup, holdout, .ntc pack (both line formats)
internal/model/       weights, quantization, integer references, accbound
internal/emit/        nnue_weights.h writers, .ntw, net.bin, legacy readers
internal/train/       online.go (draughts oracle), batch.go (production)
internal/chess/       FEN, bitboards, eval_classical, 768 features
internal/gpu/         interface, !cuda stub, cuda cgo wrapper, kernels.cu
tools/                check_gpu_writer.py (Go-vs-Python byte identity)
```

Adding a **draughts** variant is one struct literal in
`internal/variant/variant.go`. Adding another *kernel* is not — chess needed a
branch in `variant`, `corpus`, `model`, `emit`, `train` and `train_gpu.py`, plus
two new gate scripts. If a future engine shares the chess kernel it is one
struct literal again.
