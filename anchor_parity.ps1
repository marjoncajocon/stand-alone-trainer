# anchor_parity.ps1 - the gate for the chess ANCHOR.
#
# The chess net is a RESIDUE on eval_classical(), not a replacement for it
# (chess-cli/c_port/tools/trainer/README.txt:28-42). This trainer computes that
# anchor in Go (internal/chess/eval.go) so the kit stays a pure cross-compiled
# Go binary with no C toolchain in the training path. That only holds up while
# the Go copy agrees with the C EXACTLY.
#
# What this runs: internal/chess's TestAnchorParity, which compiles a fresh
# probe against the LIVE chess-cli/c_port sources and compares eval_classical
# over thousands of real corpus FENs, strided so the sample reaches endgames
# and not just openings. Plus the FEN round-trip and feature-symmetry tests.
#
# WHEN TO RUN IT: before every --pack. The anchor is frozen INTO the .ntc at
# pack time, so a drifted Go copy produces a corpus that trains perfectly on
# Colab and yields a net fitted against an anchor the engine does not compute.
# Nothing fails loudly; the net is just worse. In particular re-run this after
# tools/texel.c retunes the weights and tools/bake_weights.py bakes them back
# into eval.c / pesto_tables.h - that is the expected way for it to drift.
#
# A SKIP IS A FAILURE HERE. The Go test skips when zig or the engine sources
# are missing, which is right for `go test ./...` but wrong for a gate: a
# skipped check that reads as green is exactly the trap README.md:140-143
# records for probe_nnue.exe. So this script fails on SKIP.
#
# Keep this file pure ASCII (PowerShell 5.1 reads it as ANSI).
param(
    [string]$Go = "",
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if ($Go -eq "") {
    $cmd = Get-Command go -ErrorAction SilentlyContinue
    if ($cmd) { $Go = $cmd.Source }
}
if ($Go -eq "" -or -not (Test-Path $Go)) {
    Write-Output "ANCHOR PARITY: FAIL - go not found (pass -Go <path\to\go.exe>)"
    exit 1
}

Write-Output "regenerating nothing - this gate reads the CURRENT internal/chess"
Write-Output "against the CURRENT chess-cli/c_port sources."
Write-Output ""

$out = & $Go test ./internal/chess/ -run "TestAnchorParity|TestFENRoundTrip|TestFeatureExtraction" -v -timeout 30m 2>&1
$rc = $LASTEXITCODE
$out | ForEach-Object { Write-Output $_ }

$text = ($out | Out-String)

if ($rc -ne 0) {
    Write-Output ""
    Write-Output "ANCHOR PARITY: FAIL - DO NOT PACK A CORPUS WITH THIS BUILD"
    Write-Output "The Go eval_classical disagrees with chess-cli/c_port/eval.c."
    Write-Output "If eval.c was just retuned, re-extract pesto.go and the weight"
    Write-Output "block at the top of internal/chess/eval.go, then re-run."
    exit 1
}

if ($text -match "--- SKIP") {
    Write-Output ""
    Write-Output "ANCHOR PARITY: FAIL - the gate SKIPPED, which proves nothing."
    Write-Output "Needs zig (PATH or D:\env\zig), chess-cli/c_port sources, and a"
    Write-Output "chess corpus to sample FENs from."
    exit 1
}

Write-Output ""
Write-Output "ANCHOR PARITY: PASS"
exit 0
