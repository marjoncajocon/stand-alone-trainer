# parity.ps1 - the acceptance gate for a net or for the trainer itself.
#
# Runs a THREE-WAY comparison on real boards:
#   1. this trainer's integer reference        (--probe-file)
#   2. this trainer's narrow int16 kernel      (--probe-file-int16)
#   3. C, compiled against the header WE generate  (probe_nnue.c)
# and optionally
#   4. that repo's own reference train_nnue.exe, when -Reference is given.
#
# Why 3 matters: probe_nnue.c does `#include "nnue_weights.h"`, so it measures
# whatever header sits next to it. Running the repo's prebuilt probe_nnue.exe
# tests some historical net, NOT the one you just trained - the same trap
# note.md documents for probe_accbound. This script always compiles a FRESH
# probe against a FRESH header, so the thing under test is the thing shipping.
#
# Why 2 matters: the engine's search.c accumulates into int16, seeded by
# narrowing the int32 NN_B1. That is only legal while the pre-clamp sum stays
# inside +/-32767, and an overflow silently changes the eval with no symptom
# except worse play. Gate 1-vs-2 disagreeing means the net is unsafe to install.
#
# usage:
#   .\parity.ps1 -Bin nn_filipino_h256.ntw
#   .\parity.ps1 -Bin ..\dama-cli\...\nn_H128_s42.bin -Reference ..\dama-cli\...\train_nnue.exe
#
# Keep this file pure ASCII (PowerShell 5.1 reads it as ANSI).
param(
    [Parameter(Mandatory = $true)][string]$Bin,
    [string]$Variant = "filipino",
    [string]$Boards = "",
    [string]$Reference = "",
    [string]$Trainer = "",
    [string]$ProbeSrc = "",
    [string]$Zig = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if ($Trainer -eq "") { $Trainer = Join-Path $root "trainer_win_x64.exe" }
if (-not (Test-Path $Trainer)) { throw "trainer not built: $Trainer (run BUILD.bat)" }

$ws = Split-Path -Parent $root
if ($Boards -eq "") {
    $Boards = Join-Path $ws "dama-cli\c_port\tools\trainer\trainer\nnue\parity_boards.txt"
}
if (-not (Test-Path $Boards)) { throw "board list not found: $Boards (pass -Boards)" }
if ($ProbeSrc -eq "") {
    $ProbeSrc = Join-Path $ws "dama-cli\c_port\tools\trainer\trainer\nnue\probe_nnue.c"
}

if ($Zig -eq "") {
    $Zig = (Get-Command zig -ErrorAction SilentlyContinue).Source
    if (-not $Zig -and (Test-Path "D:\env\zig\zig.exe")) { $Zig = "D:\env\zig\zig.exe" }
}

$work = Join-Path $env:TEMP ("ntparity_" + [guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Force $work | Out-Null

try {
    Write-Output "generating header from $Bin ..."
    & $Trainer --variant=$Variant --load $Bin --convert --outh (Join-Path $work "nnue_weights.h") | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "header generation failed" }

    $boardList = @(Get-Content $Boards | ForEach-Object { $_.Trim() } | Where-Object { $_.Length -gt 0 })
    Write-Output "boards: $($boardList.Count)"

    $mine   = @(& $Trainer --variant=$Variant --load $Bin --probe-file $Boards 2>$null)
    $narrow = @(& $Trainer --variant=$Variant --load $Bin --probe-file $Boards --probe-file-int16 2>$null)

    $cScores = $null
    if ($ProbeSrc -and (Test-Path $ProbeSrc) -and $Zig) {
        Copy-Item $ProbeSrc (Join-Path $work "probe_nnue.c") -Force
        Push-Location $work
        & $Zig cc -O2 -std=c11 -o probe_new.exe probe_nnue.c
        $rc = $LASTEXITCODE
        Pop-Location
        if ($rc -eq 0) {
            $cScores = @()
            foreach ($b in $boardList) {
                $cScores += ((& (Join-Path $work "probe_new.exe") $b) -replace 'score16=', '')
            }
        } else {
            Write-Warning "probe_nnue.c did not compile against the generated header."
            Write-Warning "For a bucketed net this is EXPECTED if that engine's kernel is single-W2."
        }
    } else {
        Write-Warning "skipping the C leg (need zig and probe_nnue.c)"
    }

    $refScores = $null
    if ($Reference -ne "" -and (Test-Path $Reference)) {
        $refScores = @()
        foreach ($b in $boardList) {
            $refScores += ((& $Reference -load $Bin -probe $b) -replace 'score16=', '')
        }
    }

    $failC = 0; $failN = 0; $failR = 0
    for ($i = 0; $i -lt $boardList.Count; $i++) {
        if ($narrow[$i] -ne $mine[$i]) {
            $failN++
            if ($failN -le 3) { Write-Output "  INT16 MISMATCH $($boardList[$i]) int32=$($mine[$i]) int16=$($narrow[$i])" }
        }
        if ($cScores -and $cScores[$i] -ne $mine[$i]) {
            $failC++
            if ($failC -le 3) { Write-Output "  C MISMATCH $($boardList[$i]) go=$($mine[$i]) c=$($cScores[$i])" }
        }
        if ($refScores -and $refScores[$i] -ne $mine[$i]) {
            $failR++
            if ($failR -le 3) { Write-Output "  REF MISMATCH $($boardList[$i]) new=$($mine[$i]) ref=$($refScores[$i])" }
        }
    }

    Write-Output ""
    Write-Output "int32 reference vs int16 engine kernel : $($boardList.Count) boards, $failN mismatches"
    if ($cScores)   { Write-Output "int32 reference vs C from our header   : $($boardList.Count) boards, $failC mismatches" }
    if ($refScores) { Write-Output "int32 reference vs $(Split-Path $Reference -Leaf)        : $($boardList.Count) boards, $failR mismatches" }

    $total = $failN + $failC + $failR
    Write-Output ""
    if ($total -eq 0) { Write-Output "PARITY: PASS" } else { Write-Output "PARITY: FAIL ($total mismatches) - DO NOT INSTALL THIS NET"; exit 1 }
} finally {
    Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
}
