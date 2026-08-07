# build_all.ps1 - builds the NNUE trainer for Windows and Linux.
#
# Modelled on stand-alone-selfplay\build_all.ps1, with one big simplification:
# there is nothing to discover. The trainer is ONE binary that handles every
# variant at runtime via --variant, because unlike the selfplay kit it does not
# link any engine source - it only reads text/packed data and writes a header.
#
# Cross-compilation is free: CGO_ENABLED=0 makes the Go toolchain emit a static
# binary for any GOOS/GOARCH with no C toolchain involved. That is the whole
# reason the trainer is Go rather than C.
#
# The CUDA build is NOT produced here. It needs nvcc, and it is built natively
# on Colab by colab/build_colab.sh - see README.md.
#
# Keep this file pure ASCII: PowerShell 5.1 reads it as ANSI, so a stray em
# dash in a comment is a parser error, not a cosmetic issue.
param(
    [string[]]$Targets = @("win", "linux"),
    [switch]$Force,
    [switch]$Clean,
    [switch]$Test
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

$go = (Get-Command go -ErrorAction SilentlyContinue).Source
if (-not $go) {
    foreach ($c in @("D:\env\go_lang\go1.23.12.windows-amd64\go\bin\go.exe",
                     "C:\Go\bin\go.exe")) {
        if (Test-Path $c) { $go = $c; break }
    }
}
if (-not $go) { throw "go not found on PATH. Install Go 1.21+ or add it to PATH." }

if ($Clean) {
    Remove-Item -Force -ErrorAction SilentlyContinue trainer_win_x64.exe, trainer_linux_x64
    Remove-Item -Force -ErrorAction SilentlyContinue BUILD_MANIFEST.txt
    "cleaned"
}

# Source hash doubles as a provenance check and drives incremental builds, the
# same contract as the selfplay kit's BUILD_MANIFEST.txt.
$srcFiles = Get-ChildItem -Recurse -File -Include *.go, go.mod |
    Where-Object { $_.FullName -notlike "*\data\*" } | Sort-Object FullName
$sha = [System.Security.Cryptography.SHA256]::Create()
$buf = New-Object System.IO.MemoryStream
foreach ($f in $srcFiles) {
    $b = [System.IO.File]::ReadAllBytes($f.FullName)
    $buf.Write($b, 0, $b.Length)
}
$hash = ([BitConverter]::ToString($sha.ComputeHash($buf.ToArray())) -replace '-', '').ToLower().Substring(0, 12)
$buf.Dispose()
$date = Get-Date -Format "yyyy-MM-dd"

$manifest = "BUILD_MANIFEST.txt"
$prev = @{}
if (Test-Path $manifest) {
    foreach ($line in Get-Content $manifest) {
        if ($line -match '^(\S+)\s+(\S+)\s+(\S+)$') { $prev[$Matches[1]] = $Matches[2] }
    }
}

if ($Test) {
    "running go vet + tests..."
    & $go vet ./...
    if ($LASTEXITCODE -ne 0) { throw "go vet failed" }
    & $go test ./...
    if ($LASTEXITCODE -ne 0) { throw "go test failed" }
}

$rows = @()
foreach ($t in $Targets) {
    switch ($t) {
        "win"   { $goos = "windows"; $out = "trainer_win_x64.exe" }
        "linux" { $goos = "linux";   $out = "trainer_linux_x64" }
        default { throw "unknown target '$t' (use win or linux)" }
    }

    if (-not $Force -and $prev[$t] -eq $hash -and (Test-Path $out)) {
        "  $t : up to date ($hash)"
        $rows += "$t $hash $date"
        continue
    }

    $env:CGO_ENABLED = "0"      # pure Go: static, and cross-compiles with no C toolchain
    $env:GOOS = $goos
    $env:GOARCH = "amd64"
    # -trimpath keeps absolute paths out of the binary so two machines building
    # the same source produce the same bytes.
    & $go build -trimpath -ldflags "-s -w -X main.buildHash=$hash -X main.buildDate=$date" -o $out .\cmd\trainer
    $rc = $LASTEXITCODE
    Remove-Item Env:\GOOS, Env:\GOARCH, Env:\CGO_ENABLED -ErrorAction SilentlyContinue
    if ($rc -ne 0) { throw "build failed for $t" }

    $kb = [math]::Round((Get-Item $out).Length / 1KB)
    "  $t : built $out ($kb KB, $hash)"
    $rows += "$t $hash $date"
}

"# stand-alone-trainer build manifest - target source_hash build_date" | Set-Content $manifest -Encoding ascii
"# rewritten by build_all.ps1; delete it to force a full rebuild"     | Add-Content $manifest -Encoding ascii
$rows | Add-Content $manifest -Encoding ascii

""
"source hash $hash"
"NOTE: --gpu=1 needs the CUDA build, which is produced ON COLAB by"
"      colab/build_colab.sh. These binaries are CPU-only by design and will"
"      exit 3 if asked for a GPU."
