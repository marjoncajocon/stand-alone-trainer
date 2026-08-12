# parity_chess.ps1 - the acceptance gate for the CHESS kernel.
#
# parity.ps1 covers the six draughts engines. Chess needs its own because it is
# a different kernel: stm-relative features, one accumulator, an output bias,
# QA=255, and an anchor that is a whole hand-crafted eval rather than a piece
# count. Three legs, all compiled fresh against the LIVE chess-cli sources:
#
#   1. ANCHOR      Go eval_classical vs chess-cli/c_port/eval.c, over real
#                  corpus FENs strided into the endgame. Exact.
#   2. KERNEL      Go ScoreCP vs chess-cli/c_port/nnue.c, compiled against the
#                  header WE generate. Exact.
#   3. QUANTIZER   our net.bin -> chess-cli's own n2h.exe -> a header, which
#                  must be byte-identical to the one we write directly. One
#                  diff covers every scale (QA on w1/b1, QB on w2, QA*QB on b2)
#                  against the tool that produced every shipped header so far.
#
# There is deliberately NO int16-accumulator leg, which parity.ps1 has as its
# gate 2. chess-cli keeps its accumulator in int32 on purpose (position.h:34-42
# says pre-activation sums can exceed the int16 range with this net's weight
# magnitudes), so an int16 check here would fail good nets. Reporting it as
# "n/a" rather than quietly skipping it is the point - a skipped gate that
# reads as green is the trap README.md:140-143 records.
#
# Run this BEFORE installing any chess header, and run anchor_parity.ps1 before
# every --pack (the anchor is frozen into the .ntc at pack time).
#
# Keep this file pure ASCII (PowerShell 5.1 reads it as ANSI).
param(
    [string]$Go = "",
    [string]$Bin = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if ($Go -eq "") {
    $cmd = Get-Command go -ErrorAction SilentlyContinue
    if ($cmd) { $Go = $cmd.Source }
}
if ($Go -eq "" -or -not (Test-Path $Go)) {
    Write-Output "CHESS PARITY: FAIL - go not found (pass -Go <path\to\go.exe>)"
    exit 1
}

Write-Output "=== chess parity: anchor, kernel, quantizer ==="
Write-Output ""

$fail = 0

Write-Output "--- leg 1/3: ANCHOR (Go eval_classical vs C eval.c) ---"
$out1 = & $Go test ./internal/chess/ -run "TestAnchorParity|TestFENRoundTrip" -v -timeout 30m 2>&1
$rc1 = $LASTEXITCODE
$out1 | ForEach-Object { Write-Output $_ }
if ($rc1 -ne 0) { $fail++ }
if (($out1 | Out-String) -match "--- SKIP") {
    Write-Output "  SKIPPED - this proves nothing; needs zig, chess-cli sources and a corpus."
    $fail++
}

Write-Output ""
Write-Output "--- legs 2+3/3: KERNEL (vs nnue.c) and QUANTIZER (vs n2h.exe) ---"
$out2 = & $Go test ./internal/emit/ -run "TestChess" -v -timeout 30m 2>&1
$rc2 = $LASTEXITCODE
$out2 | ForEach-Object { Write-Output $_ }
if ($rc2 -ne 0) { $fail++ }
if (($out2 | Out-String) -match "--- SKIP") {
    Write-Output "  SKIPPED - this proves nothing; needs zig and chess-cli sources."
    $fail++
}

Write-Output ""
Write-Output "int16 narrow-accumulator leg: N/A for chess (the engine uses int32 too)"

if ($Bin -ne "") {
    if (-not (Test-Path $Bin)) {
        Write-Output "CHESS PARITY: FAIL - no such net: $Bin"
        exit 1
    }
    Write-Output ""
    Write-Output "--- generating a header from $Bin ---"
    $tmp = Join-Path $env:TEMP ("ntchess_" + [guid]::NewGuid().ToString("N").Substring(0, 8) + ".h")
    & $Go run ./cmd/trainer --variant=chess --load $Bin --convert --outh $tmp | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Output "  header generation FAILED"
        $fail++
    } else {
        Write-Output "  wrote $tmp"
        Remove-Item $tmp -ErrorAction SilentlyContinue
    }
}

Write-Output ""
if ($fail -eq 0) {
    Write-Output "CHESS PARITY: PASS"
    Write-Output ""
    Write-Output "Still required before shipping (chess-cli/c_port/tools/trainer/README.txt:70-75):"
    Write-Output "  copy nnue_weights.h into chess-cli\c_port\ and rebuild the NNUE flavors"
    Write-Output "  chess_nnue.exe nnueinc        must PASS"
    Write-Output "  chess.exe evalsym             must PASS"
    Write-Output "  tools\abmatch.exe chess_nnue.dll chess.dll 400 100 1 13    (960)"
    Write-Output "  tools\abmatch.exe chess_nnue.dll chess.dll 400 100 0 17 4  (standard)"
    Write-Output "  ship -DUSE_NNUE only if it wins BOTH (reject <48%, accept >=51.5%)"
    exit 0
}
Write-Output "CHESS PARITY: FAIL ($fail leg(s)) - DO NOT INSTALL THIS HEADER"
exit 1
