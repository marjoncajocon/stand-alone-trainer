# Where the selfplay data goes

Everything is flat. Binaries and `data/` sit together at the kit root:

```
stand-alone-trainer\
    trainer_win_x64.exe
    trainer_linux_x64
    trainer_colab            <- built on Colab, the only one with --gpu
    count_win_x64.exe        <- data inventory; count_linux_x64 too
    build_colab.sh
    parity.ps1
    data\<variant>\*.txt     <- PUT THE DATA HERE
    data\<variant>.ntc       <- or a packed corpus (wins if present)
```

No flags needed:

```powershell
.\trainer_win_x64.exe --variant=filipino --h 256
```

```bash
./trainer_colab --variant=filipino --gpu=1 --algo=minibatch --h=256 --epochs=40
```

Search order, first hit wins:

1. `<exe-dir>\data\<variant>.ntc` — packed corpus
2. `<exe-dir>\data\<variant>\*.txt`
3. `<exe-dir>\..\data\<variant>...` — dev fallback for `go run`
4. `<cwd>\data\<variant>...`

If nothing is found the error prints every path it tried. `--data` and
`--corpus` still override; `--data` is repeatable.

`count_win_x64.exe` / `count_linux_x64` use the **same** search order — that is
the point of sharing `internal/datadir` — so what the inventory reports is what
training will read.

## Counting a corpus

```powershell
.\count_win_x64.exe --variant=filipino --files
```

This supersedes the engine kits' `c_port\tools\trainer\count.ps1`, which walked a
single flat `data\*.txt` folder and counted games by grepping `done:` in the
paired `.log`. Two things changed:

**Games come from the data.** A count is the number of distinct `gameid` values
**within each file**, summed across files. `gameid` restarts at 0 in every worker
file — `stand-alone-selfplay` runs one process per worker and each writes its own
`.txt` — so the scope has to be one file, and the totals then add up with no
cross-file dedup. The `.log` `done:` grep still works as an independent check
(verified: both say 9 on the shipped filipino sample), but it is no longer the
source of truth, so a corpus without logs is still countable.

Where they can legitimately disagree: a game that produced zero quiet positions
writes no `.txt` lines at all (count < log), and a killed run writes lines for a
game that never logged `done:` (count > log).

**Malformed and TB-skipped are not the same thing.** They sit on opposite sides
of the loader's line counter:

- `MALFORMED` is rejected *before* the loader counts it, so it is **not** in
  `LINES`. Draughts: fewer than 3 fields, or a board whose length is not the
  variant's cell count. Chess: fewer than 8 fields, an unparseable FEN, or a
  result token that is not `1-0`/`0-1`/`1/2-1/2`. A wall of these almost always
  means the wrong `--variant` for the geometry on disk (64 vs 100 cells).
- `TB-SKIP` **is** in `LINES` and is only excluded from `UNIQUE`: the position is
  well-formed, it is just in that variant's tablebase territory, which its
  reference trainer drops. `--list-variants` prints the per-variant rule.

A **packed `.ntc`** is read header-only, so `GAMES` and `MALFORMED` print `-`:
the format has never carried a game count, and the packer dropped bad lines
before writing. `--data` at the `.txt` files answers both.

## Chess on Colab — the whole loop

Upload three things to Colab: `trainer_linux_x64`, `train_gpu.py`, and your
chess `.txt` data. Then:

```bash
# 0. once per session
chmod +x trainer_linux_x64

# 1. pack. The Go binary finds BOTH data/chess and data/chess960 by itself.
./trainer_linux_x64 --variant=chess --pack --out-corpus chess.ntc

# 2. train.  --h 256 is REQUIRED (the engine static-asserts it).
#    K=1/400 comes from the .ntc; you do not pass it.
python3 train_gpu.py --corpus chess.ntc --h 256 --epochs 40 \
    --out nn_chess_h256.ntw --log train_chess.log
```

Pack **on Colab** rather than at home if your machine is short on RAM: the full
918 MB corpus needs a few GB to dedup, and Colab has ~12 GB.

Then download `nn_chess_h256.ntw` + `train_chess.log` and finish at home — the
header is never installed from a run that was not re-checked locally:

```powershell
.\trainer_win_x64.exe --variant=chess --load nn_chess_h256.ntw --convert --outh nnue_weights.h
.\parity_chess.ps1 -Bin nn_chess_h256.ntw
```

`train_gpu.py` can also pack for you, and `--data` is repeatable because chess
pools two folders:

```bash
python3 train_gpu.py --variant chess \
    --data "data/chess/*.txt" --data "data/chess960/*.txt" \
    --keep-corpus chess.ntc --h 256 --epochs 40 --out nn_chess_h256.ntw
```

## Chess reads TWO folders

`--variant=chess` globs **both**:

```
data\chess\*.txt        standard
data\chess960\*.txt     Chess960
```

They pool into one net, which is what `chess-cli`'s own kit does — the eval is
position-agnostic and the app ships a single net. The folders stay separate on
disk because `stand-alone-selfplay` writes them that way, and because the
holdout group key uses the **folder name**: two worker files with the same
basename, one standard and one 960, must not land in the same holdout group.

Chess lines are a different shape from draughts — the FEN spans six
whitespace-separated fields:

```
<fen> <result> <gameid> [<eval>]
rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2 1/2-1/2 0 34
```

`<result>` and `<eval>` are **White-perspective**; the trainer flips both to the
side to move, because the net's output and the `eval_classical` anchor are
stm-relative. Two positions differing only in the halfmove/fullmove clocks
dedup together; differing in side to move, castling rights, en-passant file or
(for 960) castling rook files do not.

Chess also pins `--h 256` and uses `K = 1/400` (draughts uses 1/256). The K
travels inside the `.ntc`, so `train_gpu.py` picks it up with no flag.

`data/` is gitignored — the filipino corpus alone is 2.9 GB and the 1M-game
target is roughly 5.5 GB.

## The holdout split is portable

The train/valid split is `FNV1a(<group key> + ":" + gameid) % 20`, applied to
whole GAMES so correlated positions never straddle it.

The group key defaults to **`rel`** = `<variant>/<basename>` — identical on
Windows and Linux, and unaffected by where the kit lives. Verified: an absolute
glob and `*.txt` from inside the folder both give 327,837 train / 17,497 valid.

| `--group-key` | hashes | use when |
|---|---|---|
| `rel` *(default)* | `filipino/selfplay_x.txt` | always |
| `path` | `D:\...\data\filipino\selfplay_x.txt` | reproducing a historical reference run — **not portable**, `data\x.txt` and `data/x.txt` hash differently |
| `basename` | `selfplay_x.txt` | legacy |

Changing this changes which games are held out, so MSE from two different
settings cannot be compared.

## Pack before going to Colab

```powershell
.\trainer_win_x64.exe --variant=filipino --pack --out-corpus data\filipino.ntc
```

~10x smaller than the text (556K lines -> 14 MB), loads instantly, and freezes
the split inside the file so a Colab run is directly comparable to a local one.
Round-trip is exact: text and pack give identical loader counters and identical
epoch-1 MSE.

For chess, **run `anchor_parity.ps1` first, every time**. `--pack` computes
`eval_classical` per position and freezes it into the `.ntc`; if the Go copy of
that eval has drifted from `chess-cli/c_port/eval.c`, the corpus trains
perfectly on Colab and produces a net fitted against an anchor the engine does
not compute. Nothing fails loudly — the net is just worse. Drift is expected
after a texel retune, because `tools/bake_weights.py` rewrites `eval.c`.

The pack format is `NTC2` (it carries the kernel and the K scale so
`train_gpu.py` needs no variant knowledge). `NTC1` files still load; chess has
never had one.

## Keep the .log files

The selfplay kit writes a `.log` beside every `.txt`. The `*.txt` glob skips
them, so they cost nothing — and their header carries `engine_hash=` and
`depth=`, the only provenance a corpus has. They also remain a useful independent
check on `count`'s `GAMES` column:

```powershell
$g = 0; Get-ChildItem "data\filipino\*.log" |
    ForEach-Object { $g += (Select-String -Path $_.FullName -Pattern "done:").Count }; $g
```
