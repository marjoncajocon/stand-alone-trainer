# build_all.ps1 - builds the NNUE trainer and the data counter for Windows and
# Linux.
#
# Modelled on stand-alone-selfplay\build_all.ps1, with one big simplification:
# there is nothing to discover. Each binary handles every variant at runtime via
# --variant, because unlike the selfplay kit neither links any engine source -
# they only read text/packed data and write a header.
#
# Two binaries ship from this one module and one source hash: trainer (trains and
# emits nnue_weights.h) and count (reports games/positions per variant). They
# share internal/corpus, so the counter's numbers are the loader's numbers.
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
    Remove-Item -Force -ErrorAction SilentlyContinue trainer_win_x64.exe, trainer_linux_x64, count_win_x64.exe, count_linux_x64
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

# One entry per shipped binary. The output name is derived, not listed, so the
# existing names (trainer_win_x64.exe, trainer_linux_x64) are reproduced exactly.
$bins = @(
    @{ Name = "trainer"; Pkg = ".\cmd\trainer" },
    @{ Name = "count";   Pkg = ".\cmd\count"   }
)

$rows = @()
foreach ($t in $Targets) {
    switch ($t) {
        "win"   { $goos = "windows"; $ext = ".exe" }
        "linux" { $goos = "linux";   $ext = ""     }
        default { throw "unknown target '$t' (use win or linux)" }
    }

    foreach ($b in $bins) {
        $out = "$($b.Name)_$($t)_x64$ext"
        # The manifest key is <target>-<binary>, not <target>: with one row per
        # target, a missing count binary next to a current trainer would report
        # "up to date" and skip a build that has no output file.
        $key = "$t-$($b.Name)"

        if (-not $Force -and $prev[$key] -eq $hash -and (Test-Path $out)) {
            "  $key : up to date ($hash)"
            $rows += "$key $hash $date"
            continue
        }

        $env:CGO_ENABLED = "0"      # pure Go: static, and cross-compiles with no C toolchain
        $env:GOOS = $goos
        $env:GOARCH = "amd64"
        # -trimpath keeps absolute paths out of the binary so two machines building
        # the same source produce the same bytes. -X needs main.buildHash to be
        # DECLARED in the package being built, which both mains now do.
        & $go build -trimpath -ldflags "-s -w -X main.buildHash=$hash -X main.buildDate=$date" -o $out $b.Pkg
        $rc = $LASTEXITCODE
        Remove-Item Env:\GOOS, Env:\GOARCH, Env:\CGO_ENABLED -ErrorAction SilentlyContinue
        if ($rc -ne 0) { throw "build failed for $key" }

        $kb = [math]::Round((Get-Item $out).Length / 1KB)
        "  $key : built $out ($kb KB, $hash)"
        $rows += "$key $hash $date"
    }
}

"# stand-alone-trainer build manifest - target-binary source_hash build_date" | Set-Content $manifest -Encoding ascii
"# rewritten by build_all.ps1; delete it to force a full rebuild"             | Add-Content $manifest -Encoding ascii
"# the key gained a -binary suffix when count shipped, so the first run after" | Add-Content $manifest -Encoding ascii
"# that change rebuilds everything once; that is expected, not a bug"         | Add-Content $manifest -Encoding ascii
$rows | Add-Content $manifest -Encoding ascii

""
"source hash $hash"
"Data inventory: count_win_x64.exe --files   (or COUNT.bat)"
"NOTE: --gpu=1 needs the CUDA build, which is produced ON COLAB by"
"      colab/build_colab.sh. These binaries are CPU-only by design and will"
"      exit 3 if asked for a GPU."
