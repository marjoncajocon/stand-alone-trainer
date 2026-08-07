#!/usr/bin/env bash
# build_colab.sh - build the CUDA-enabled trainer, natively, on Colab.
#
# This is the ONLY build that uses cgo, and it happens on the machine that will
# run it, so cross-compilation is irrelevant here. Every build that ships from
# the Windows box stays pure Go (CGO_ENABLED=0) and cross-compiles freely.
#
#   bash build_colab.sh
#
# Produces ./trainer_colab, which supports --gpu=1. Takes ~30 s.
set -euo pipefail

cd "$(dirname "$0")"

# Source sanity check. The usual cause of a miss here is a zip made by Windows
# PowerShell Compress-Archive: it writes BACKSLASH path separators, which Linux
# unzip treats as part of the filename rather than as directories, so the tree
# arrives flattened into files literally named 'internal\gpu\kernels.cu'.
if [ ! -f internal/gpu/kernels.cu ]; then
  echo "error: internal/gpu/kernels.cu not found (cwd: $PWD)" >&2
  if ls -1 2>/dev/null | grep -q '\\'; then
    echo >&2
    echo "  Found files with literal backslashes in their names - this archive was" >&2
    echo "  made by Windows Compress-Archive and unzipped flat. Repair it in place:" >&2
    echo >&2
    echo "    python3 - <<'EOF'" >&2
    echo "    import os, shutil" >&2
    echo "    for f in os.listdir('.'):" >&2
    echo "        if '\\\\' in f:" >&2
    echo "            t = f.replace('\\\\', '/')" >&2
    echo "            os.makedirs(os.path.dirname(t), exist_ok=True)" >&2
    echo "            shutil.move(f, t)" >&2
    echo "    EOF" >&2
    echo >&2
    echo "  Better: upload stand-alone-trainer.tar.gz instead and" >&2
    echo "    tar xzf stand-alone-trainer.tar.gz" >&2
  else
    echo "  Run this from the kit root, where go.mod lives." >&2
  fi
  exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "error: nvcc not found." >&2
  echo "  Runtime > Change runtime type > T4 GPU, then rerun this." >&2
  echo "  (Changing runtime type RESTARTS the VM and wipes /content.)" >&2
  echo "  Without a GPU, build the CPU trainer instead:" >&2
  echo "    CGO_ENABLED=0 go build -o trainer_linux_x64 ./cmd/trainer" >&2
  exit 1
fi

# Go: Colab images ship no Go at all, and `apt-get install golang-go` on the
# Ubuntu 22.04 image gives 1.18 - older than this module needs, so it fails with
# "go.mod requires go >= 1.21". Install the official tarball instead (~15 s).
GO_VER=1.23.5
if ! command -v go >/dev/null 2>&1 || \
   [ "$(go env GOVERSION 2>/dev/null | sed 's/go//' | cut -d. -f2)" -lt 21 ] 2>/dev/null; then
  echo "==> installing Go ${GO_VER}"
  curl -fsSL "https://go.dev/dl/go${GO_VER}.linux-amd64.tar.gz" -o /tmp/go.tgz
  rm -rf /usr/local/go && tar -C /usr/local -xzf /tmp/go.tgz
  export PATH=/usr/local/go/bin:$PATH
fi
export PATH=/usr/local/go/bin:$PATH
go version

# Always start from a clean object dir. A stale .build/ from a failed run is a
# real hazard: `ar r` replaces a member but the archive symbol index can end up
# describing the OLD object, which shows up as exactly one symbol failing to
# link while its neighbours resolve fine.
rm -rf .build
mkdir -p .build

# One .cu, compiled for every architecture Colab hands out: P100 (sm_60),
# V100 (70), T4 (75), A100 (80), L4/RTX (89). Listing them all avoids a JIT
# stall on first launch and makes the binary portable across session types.
# -Wno-deprecated-gpu-targets silences the sm_60/sm_70 deprecation notice.
echo "==> nvcc"
nvcc -O3 -std=c++14 -fmad=false -Wno-deprecated-gpu-targets \
     -gencode arch=compute_60,code=sm_60 \
     -gencode arch=compute_70,code=sm_70 \
     -gencode arch=compute_75,code=sm_75 \
     -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_86,code=sm_86 \
     -gencode arch=compute_89,code=sm_89 \
     -c internal/gpu/kernels.cu -o .build/kernels.o

# -fmad=false is load-bearing, not an optimization preference: it stops the
# compiler fusing a*b+c into an FMA, which would change rounding and break the
# CPU/GPU byte-identity gate (see README).

# Verify every symbol cgo expects is present AND unmangled before wasting a link.
# nvcc compiles .cu as C++, so anything accidentally left outside the extern "C"
# block would appear here as a mangled _Z... name and fail to link later with a
# confusing message.
echo "==> checking exported symbols"
missing=0
for sym in nt_gpu_error nt_gpu_available nt_gpu_name nt_gpu_upload \
           nt_gpu_step nt_gpu_valid_mse nt_gpu_download nt_gpu_close; do
  if nm -g --defined-only .build/kernels.o | grep -qw "$sym"; then
    printf '    %-18s ok\n' "$sym"
  else
    printf '    %-18s MISSING\n' "$sym"
    missing=1
  fi
done
if [ "$missing" = 1 ]; then
  echo >&2
  echo "error: kernels.o does not export the symbols cgo needs." >&2
  echo "  Any name shown MISSING is defined outside the extern \"C\" block in" >&2
  echo "  internal/gpu/kernels.cu, so nvcc gave it C++ mangling. Mangled names" >&2
  echo "  present in the object:" >&2
  nm -g --defined-only .build/kernels.o | grep ' T _Z' >&2 || true
  exit 1
fi

echo "==> go build -tags cuda"
CUDA_LIB="$(dirname "$(command -v nvcc)")/../lib64"
# Link the object DIRECTLY rather than via `ar` - no archive, so no archive
# symbol index to go stale, and no -l ordering subtleties.
# libcudart is linked STATICALLY so the result depends only on the NVIDIA driver
# (libcuda.so.1), which is present on any machine that has a GPU at all. That
# makes trainer_colab a self-contained binary you can keep and reuse.
CGO_ENABLED=1 \
CGO_CFLAGS="-I$(dirname "$(command -v nvcc)")/../include" \
CGO_LDFLAGS="$PWD/.build/kernels.o -L$CUDA_LIB -lcudart_static -lstdc++ -lm -ldl -lrt -lpthread" \
  go build -tags cuda -o trainer_colab ./cmd/trainer || {
    echo >&2
    echo "static libcudart failed; retrying against the shared one" >&2
    CGO_ENABLED=1 \
    CGO_CFLAGS="-I$(dirname "$(command -v nvcc)")/../include" \
    CGO_LDFLAGS="$PWD/.build/kernels.o -L$CUDA_LIB -lcudart -lstdc++ -lm" \
      go build -tags cuda -o trainer_colab ./cmd/trainer
  }

echo
echo "built ./trainer_colab"
./trainer_colab --list-variants >/dev/null && echo "smoke test OK"
echo
echo "Put data in  data/<variant>/*.txt  (or data/<variant>.ntc), then:"
echo "  ./trainer_colab --variant=filipino --gpu=1 --algo=minibatch --h=256 --epochs=40 --log=train.log"
