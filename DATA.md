# Where the selfplay data goes

Everything is flat. Binaries and `data/` sit together at the kit root:

```
stand-alone-trainer\
    trainer_win_x64.exe
    trainer_linux_x64
    trainer_colab            <- built on Colab, the only one with --gpu
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

## Keep the .log files

The selfplay kit writes a `.log` beside every `.txt`. The `*.txt` glob skips
them, so they cost nothing — and their header carries `engine_hash=` and
`depth=`, the only provenance a corpus has.
