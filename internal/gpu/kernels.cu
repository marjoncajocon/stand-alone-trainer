// kernels.cu — CUDA backend for the minibatch NNUE trainer.
//
// Compiled ONLY on Colab (see colab/build_colab.sh); never touched by the
// pure-Go cross-compiled build.
//
// The computation must match internal/train/batch.go's accumulate() exactly.
// Two design choices exist purely to make that a testable claim rather than an
// aspiration:
//
//  1. sigmoidf/expf below are a character-for-character port of the Go
//     versions, in explicit float32 ops with no libm call. Built with
//     -fmad=false so the compiler cannot fuse a*b+c into an FMA and change the
//     rounding.
//  2. The W1 gradient uses an INVERTED INDEX, not atomics. A naive
//     atomicAdd would issue batch*~14*2*H adds onto only NumFeat distinct rows
//     — brutal contention, and non-deterministic in float. Instead each step
//     counting-sorts its (feature, sample) pairs into per-feature lists (a
//     stable, fully deterministic sort over <= NumFeat buckets) and one block
//     per (feature, j-tile) walks its list in ascending sample order. Fixed
//     reduction order, no atomics, and far better memory behaviour.
//
// The acceptance gate is therefore byte equality: run --algo=minibatch
// --max-steps=200 on CPU with --threads 1 and --threads 8, and on GPU, and the
// three weight checkpoints must be identical.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CUDA_OK(call)                                                          \
  do {                                                                         \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
      snprintf(g_err, sizeof(g_err), "%s:%d %s", __FILE__, __LINE__,            \
               cudaGetErrorString(_e));                                         \
      return -1;                                                                \
    }                                                                          \
  } while (0)

static char g_err[512];

/* nt_gpu_error is defined at the bottom, INSIDE the extern "C" block. It must
 * not be defined here: nvcc compiles this file as C++, so a definition outside
 * extern "C" gets C++ name mangling and cgo's reference to the plain C symbol
 * fails to link. */

// ── device-side float32 math, mirroring internal/train/batch.go ─────────────

__device__ __forceinline__ float nt_ldexpf(float x, int k) {
  if (k > 127) {
    x *= __int_as_float((unsigned)(127 + 127) << 23);
    k -= 127;
    if (k > 127) k = 127;
  } else if (k < -126) {
    x *= __int_as_float((unsigned)(-126 + 127) << 23);
    k += 126;
    if (k < -126) k = -126;
  }
  return x * __int_as_float((unsigned)(k + 127) << 23);
}

__device__ __forceinline__ float nt_expf(float x) {
  if (x >= 88.7f) return __int_as_float(0x7f800000); // +inf
  if (x <= -87.4f) return 0.0f;
  const float log2e = 1.4426950408889634f;
  const float ln2hi = 0.693359375f;
  const float ln2lo = -2.1219444e-4f;
  float kf = x * log2e;
  int k = (kf >= 0.0f) ? (int)(kf + 0.5f) : (int)(kf - 0.5f);
  float fk = (float)k;
  float r = (x - fk * ln2hi) - fk * ln2lo;
  float p = 1.0f / 720.0f;
  p = p * r + 1.0f / 120.0f;
  p = p * r + 1.0f / 24.0f;
  p = p * r + 1.0f / 6.0f;
  p = p * r + 0.5f;
  p = p * r + 1.0f;
  p = p * r + 1.0f;
  return nt_ldexpf(p, k);
}

__device__ __forceinline__ float nt_sigmoidf(float x) {
  return 1.0f / (1.0f + nt_expf(-x));
}

__device__ __forceinline__ float nt_clip01(float x) {
  return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
}

// ── persistent device state ─────────────────────────────────────────────────

typedef struct {
  int H, numFeat, buckets;
  int nTrain, nValid, poolLen;
  int batch;
  float k;
  int isAdam;

  unsigned short *pool;
  int *off, *end;
  short *anchor;
  unsigned char *bucket;
  float *y;
  unsigned short *mirror;

  int vOff0; // index where the valid split begins (train samples come first)

  float *w1, *b1, *w2;
  float *g1, *gb, *g2;
  float *m1, *mb, *m2;
  float *v1, *vb, *v2;

  int *perm;      // this step's sample ids
  float *actA;    // batch*H crelu(acc1)
  float *actB;    // batch*H crelu(acc2)
  float *gate;    // batch*H bitmask packed as float pairs (gate1, gate2)
  float *gateB;
  float *gr;      // batch

  // inverted index over (feature -> sample slots) for the W1 gradient
  int *featCount, *featStart, *featSlotSample, *featSlotSide;
  int maxSlots;

  long long step;
} NtCtx;

static NtCtx C;

// ── kernels ─────────────────────────────────────────────────────────────────

// One block per sample, H threads per block.
__global__ void k_forward(NtCtx c, int n) {
  int s = blockIdx.x;
  if (s >= n) return;
  int j = threadIdx.x;
  int id = c.perm[s];

  float a1 = c.b1[j], a2 = c.b1[j];
  int lo = c.off[id], hi = c.end[id];
  for (int t = lo; t < hi; t++) {
    unsigned short f = c.pool[t];
    a1 += c.w1[(int)f * c.H + j];
    a2 += c.w1[(int)c.mirror[f] * c.H + j];
  }
  int wb = (c.buckets > 1) ? (int)c.bucket[id] * c.H : 0;

  float ca1 = nt_clip01(a1), ca2 = nt_clip01(a2);
  c.actA[(size_t)s * c.H + j] = ca1;
  c.actB[(size_t)s * c.H + j] = ca2;
  c.gate[(size_t)s * c.H + j] = (a1 > 0.0f && a1 < 1.0f) ? 1.0f : 0.0f;
  c.gateB[(size_t)s * c.H + j] = (a2 > 0.0f && a2 < 1.0f) ? 1.0f : 0.0f;

  // Fixed-shape tree reduction of sum_j w2[wb+j]*(ca1-ca2).
  extern __shared__ float sm[];
  sm[j] = c.w2[wb + j] * (ca1 - ca2);
  __syncthreads();
  for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
    if (j < s2) sm[j] += sm[j + s2];
    __syncthreads();
  }
  if (j == 0) {
    float net = 64.0f * sm[0];
    float p = nt_sigmoidf(c.k * ((float)c.anchor[id] + net));
    c.gr[s] = -2.0f * (c.y[id] - p) * p * (1.0f - p) * c.k * 64.0f;
  }
}

// W2 and B1 gradients: one block per hidden unit j, threads stride the batch.
// Reduced in ascending sample order inside each block.
__global__ void k_bwd_dense(NtCtx c, int n) {
  int j = blockIdx.x;
  if (j >= c.H) return;

  extern __shared__ float sm[];
  float *sw2 = sm;                 // buckets entries
  float *sb1 = sm + c.buckets;     // 1 entry

  for (int b = threadIdx.x; b < c.buckets; b += blockDim.x) sw2[b] = 0.0f;
  if (threadIdx.x == 0) sb1[0] = 0.0f;
  __syncthreads();

  // Serial over the batch in ONE thread keeps the summation order fixed and
  // identical to the CPU chunk order. H blocks already saturate the device.
  if (threadIdx.x == 0) {
    for (int s = 0; s < n; s++) {
      int id = c.perm[s];
      int wb = (c.buckets > 1) ? (int)c.bucket[id] : 0;
      float gr = c.gr[s];
      float a1 = c.actA[(size_t)s * c.H + j];
      float a2 = c.actB[(size_t)s * c.H + j];
      sw2[wb] += gr * (a1 - a2);

      float w2v = c.w2[wb * c.H + j];
      float d1 = c.gate[(size_t)s * c.H + j] * gr * w2v;
      float d2 = -c.gateB[(size_t)s * c.H + j] * gr * w2v;
      sb1[0] += d1 + d2;
    }
    for (int b = 0; b < c.buckets; b++) c.g2[b * c.H + j] = sw2[b];
    c.gb[j] = sb1[0];
  }
}

// W1 gradient via the inverted index. One block per (feature, j) pair is too
// many blocks; instead one block per feature, H threads, walking that feature's
// slot list in ascending slot order.
__global__ void k_bwd_w1(NtCtx c) {
  int f = blockIdx.x;
  if (f >= c.numFeat) return;
  int j = threadIdx.x;

  float acc = 0.0f;
  int lo = c.featStart[f], hi = c.featStart[f] + c.featCount[f];
  for (int t = lo; t < hi; t++) {
    int s = c.featSlotSample[t];
    int side = c.featSlotSide[t]; // 0 = direct (d1), 1 = mirrored (d2)
    int id = c.perm[s];
    int wb = (c.buckets > 1) ? (int)c.bucket[id] * c.H : 0;
    float gr = c.gr[s];
    float w2v = c.w2[wb + j];
    if (side == 0) {
      acc += c.gate[(size_t)s * c.H + j] * gr * w2v;
    } else {
      acc += -c.gateB[(size_t)s * c.H + j] * gr * w2v;
    }
  }
  c.g1[(size_t)f * c.H + j] = acc;
}

__global__ void k_adam(float *w, float *g, float *m, float *v, int n, float lr,
                       float scale, float bc1, float bc2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float gi = g[i] * scale;
  m[i] = 0.9f * m[i] + 0.1f * gi;
  v[i] = 0.999f * v[i] + 0.001f * gi * gi;
  float mh = m[i] / bc1;
  float vh = v[i] / bc2;
  w[i] -= lr * mh / (sqrtf(vh) + 1e-8f);
}

__global__ void k_adagrad(float *w, float *g, float *acc, int n, float lr,
                          float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float gi = g[i] * scale;
  acc[i] += gi * gi;
  w[i] -= lr * gi / sqrtf(acc[i] + 1e-8f);
}

// Held-out MSE over the valid split (which is stored after the train split).
__global__ void k_valid(NtCtx c, float *partial) {
  int s = blockIdx.x;
  if (s >= c.nValid) return;
  int j = threadIdx.x;
  int id = c.vOff0 + s;

  float a1 = c.b1[j], a2 = c.b1[j];
  for (int t = c.off[id]; t < c.end[id]; t++) {
    unsigned short f = c.pool[t];
    a1 += c.w1[(int)f * c.H + j];
    a2 += c.w1[(int)c.mirror[f] * c.H + j];
  }
  int wb = (c.buckets > 1) ? (int)c.bucket[id] * c.H : 0;

  extern __shared__ float sm[];
  sm[j] = c.w2[wb + j] * (nt_clip01(a1) - nt_clip01(a2));
  __syncthreads();
  for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
    if (j < s2) sm[j] += sm[j + s2];
    __syncthreads();
  }
  if (j == 0) {
    float p = nt_sigmoidf(c.k * ((float)c.anchor[id] + 64.0f * sm[0]));
    float d = c.y[id] - p;
    partial[s] = d * d;
  }
}

// ── host API (called from cuda.go via cgo) ──────────────────────────────────

extern "C" {

const char *nt_gpu_error(void) { return g_err; }

int nt_gpu_available(void) {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
  return n;
}

int nt_gpu_name(int ordinal, char *out, int cap) {
  cudaDeviceProp p;
  if (cudaGetDeviceProperties(&p, ordinal) != cudaSuccess) return -1;
  snprintf(out, cap, "%s (sm_%d%d, %.1f GB)", p.name, p.major, p.minor,
           (double)p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  return 0;
}

int nt_gpu_upload(int ordinal, int H, int numFeat, int buckets, int batch,
                  float k, int isAdam, const unsigned short *pool, int poolLen,
                  const int *off, const int *end, const short *anchor,
                  const unsigned char *bucket, const float *y, int nTotal,
                  int nTrain, int nValid, const unsigned short *mirror,
                  const float *w1, const float *b1, const float *w2) {
  memset(&C, 0, sizeof(C));
  CUDA_OK(cudaSetDevice(ordinal));
  C.H = H; C.numFeat = numFeat; C.buckets = buckets; C.batch = batch;
  C.k = k; C.isAdam = isAdam;
  C.nTrain = nTrain; C.nValid = nValid; C.poolLen = poolLen;
  C.vOff0 = nTrain;
  C.step = 0;

  size_t n1 = (size_t)numFeat * H, nb = (size_t)H, n2 = (size_t)buckets * H;

#define ALLOC(p, bytes) CUDA_OK(cudaMalloc((void **)&(p), (bytes)))
#define UP(p, src, bytes) CUDA_OK(cudaMemcpy((p), (src), (bytes), cudaMemcpyHostToDevice))

  ALLOC(C.pool, poolLen * sizeof(unsigned short));
  UP(C.pool, pool, poolLen * sizeof(unsigned short));
  ALLOC(C.off, nTotal * sizeof(int));   UP(C.off, off, nTotal * sizeof(int));
  ALLOC(C.end, nTotal * sizeof(int));   UP(C.end, end, nTotal * sizeof(int));
  ALLOC(C.anchor, nTotal * sizeof(short)); UP(C.anchor, anchor, nTotal * sizeof(short));
  ALLOC(C.bucket, nTotal); UP(C.bucket, bucket, nTotal);
  ALLOC(C.y, nTotal * sizeof(float)); UP(C.y, y, nTotal * sizeof(float));
  ALLOC(C.mirror, numFeat * sizeof(unsigned short));
  UP(C.mirror, mirror, numFeat * sizeof(unsigned short));

  ALLOC(C.w1, n1 * 4); UP(C.w1, w1, n1 * 4);
  ALLOC(C.b1, nb * 4); UP(C.b1, b1, nb * 4);
  ALLOC(C.w2, n2 * 4); UP(C.w2, w2, n2 * 4);
  ALLOC(C.g1, n1 * 4); ALLOC(C.gb, nb * 4); ALLOC(C.g2, n2 * 4);
  ALLOC(C.m1, n1 * 4); ALLOC(C.mb, nb * 4); ALLOC(C.m2, n2 * 4);
  ALLOC(C.v1, n1 * 4); ALLOC(C.vb, nb * 4); ALLOC(C.v2, n2 * 4);
  CUDA_OK(cudaMemset(C.m1, 0, n1 * 4)); CUDA_OK(cudaMemset(C.v1, 0, n1 * 4));
  CUDA_OK(cudaMemset(C.mb, 0, nb * 4)); CUDA_OK(cudaMemset(C.vb, 0, nb * 4));
  CUDA_OK(cudaMemset(C.m2, 0, n2 * 4)); CUDA_OK(cudaMemset(C.v2, 0, n2 * 4));

  ALLOC(C.perm, batch * sizeof(int));
  ALLOC(C.actA, (size_t)batch * H * 4);
  ALLOC(C.actB, (size_t)batch * H * 4);
  ALLOC(C.gate, (size_t)batch * H * 4);
  ALLOC(C.gateB, (size_t)batch * H * 4);
  ALLOC(C.gr, batch * 4);

  C.maxSlots = batch * 64 * 2; // generous: >= 2 * batch * max pieces
  ALLOC(C.featCount, numFeat * sizeof(int));
  ALLOC(C.featStart, numFeat * sizeof(int));
  ALLOC(C.featSlotSample, C.maxSlots * sizeof(int));
  ALLOC(C.featSlotSide, C.maxSlots * sizeof(int));
#undef ALLOC
#undef UP
  return 0;
}

// nt_gpu_step runs ONE optimizer step. The inverted index is built on the host
// (it is a counting sort over <= numFeat buckets and costs microseconds) so its
// order is trivially deterministic and identical to the CPU path.
// bc1/bc2 are Adam's bias-correction terms. They are computed BY THE CALLER in
// float64 (Go's math.Pow), exactly as internal/train/batch.go does, and passed
// in — rather than recomputed here with powf. Two reasons: a float32 powf would
// not match the CPU path bit-for-bit and would break the byte-identity gate,
// and host-side powf drags libm into the link for no benefit.
int nt_gpu_step(const int *permHost, int n, const int *slotSample,
                const int *slotSide, const int *featCount, const int *featStart,
                int nSlots, float lr, float bc1, float bc2) {
  CUDA_OK(cudaMemcpy(C.perm, permHost, n * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(C.featCount, featCount, C.numFeat * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(C.featStart, featStart, C.numFeat * sizeof(int), cudaMemcpyHostToDevice));
  if (nSlots > C.maxSlots) {
    snprintf(g_err, sizeof(g_err), "inverted index overflow: %d > %d", nSlots, C.maxSlots);
    return -1;
  }
  CUDA_OK(cudaMemcpy(C.featSlotSample, slotSample, nSlots * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(C.featSlotSide, slotSide, nSlots * sizeof(int), cudaMemcpyHostToDevice));

  size_t shm = (size_t)C.H * sizeof(float);
  k_forward<<<n, C.H, shm>>>(C, n);
  CUDA_OK(cudaGetLastError());

  size_t shm2 = (size_t)(C.buckets + 1) * sizeof(float);
  k_bwd_dense<<<C.H, 32, shm2>>>(C, n);
  CUDA_OK(cudaGetLastError());

  k_bwd_w1<<<C.numFeat, C.H>>>(C);
  CUDA_OK(cudaGetLastError());

  C.step++;
  float scale = 1.0f / (float)n;
  size_t n1 = (size_t)C.numFeat * C.H, nb = (size_t)C.H, n2 = (size_t)C.buckets * C.H;
  int tpb = 256;
  if (C.isAdam) {
    k_adam<<<(int)((n1 + tpb - 1) / tpb), tpb>>>(C.w1, C.g1, C.m1, C.v1, (int)n1, lr, scale, bc1, bc2);
    k_adam<<<(int)((nb + tpb - 1) / tpb), tpb>>>(C.b1, C.gb, C.mb, C.vb, (int)nb, lr, scale, bc1, bc2);
    k_adam<<<(int)((n2 + tpb - 1) / tpb), tpb>>>(C.w2, C.g2, C.m2, C.v2, (int)n2, lr, scale, bc1, bc2);
  } else {
    k_adagrad<<<(int)((n1 + tpb - 1) / tpb), tpb>>>(C.w1, C.g1, C.v1, (int)n1, lr, scale);
    k_adagrad<<<(int)((nb + tpb - 1) / tpb), tpb>>>(C.b1, C.gb, C.vb, (int)nb, lr, scale);
    k_adagrad<<<(int)((n2 + tpb - 1) / tpb), tpb>>>(C.w2, C.g2, C.v2, (int)n2, lr, scale);
  }
  CUDA_OK(cudaGetLastError());
  CUDA_OK(cudaDeviceSynchronize());
  return 0;
}

int nt_gpu_valid_mse(double *out) {
  float *partial = NULL;
  CUDA_OK(cudaMalloc((void **)&partial, (size_t)C.nValid * 4));
  size_t shm = (size_t)C.H * sizeof(float);
  k_valid<<<C.nValid, C.H, shm>>>(C, partial);
  CUDA_OK(cudaGetLastError());
  float *host = (float *)malloc((size_t)C.nValid * 4);
  CUDA_OK(cudaMemcpy(host, partial, (size_t)C.nValid * 4, cudaMemcpyDeviceToHost));
  double e = 0;
  for (int i = 0; i < C.nValid; i++) e += host[i];
  free(host);
  cudaFree(partial);
  *out = e / (double)C.nValid;
  return 0;
}

int nt_gpu_download(float *w1, float *b1, float *w2) {
  size_t n1 = (size_t)C.numFeat * C.H, nb = (size_t)C.H, n2 = (size_t)C.buckets * C.H;
  CUDA_OK(cudaMemcpy(w1, C.w1, n1 * 4, cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(b1, C.b1, nb * 4, cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(w2, C.w2, n2 * 4, cudaMemcpyDeviceToHost));
  return 0;
}

void nt_gpu_close(void) {
  cudaFree(C.pool); cudaFree(C.off); cudaFree(C.end); cudaFree(C.anchor);
  cudaFree(C.bucket); cudaFree(C.y); cudaFree(C.mirror);
  cudaFree(C.w1); cudaFree(C.b1); cudaFree(C.w2);
  cudaFree(C.g1); cudaFree(C.gb); cudaFree(C.g2);
  cudaFree(C.m1); cudaFree(C.mb); cudaFree(C.m2);
  cudaFree(C.v1); cudaFree(C.vb); cudaFree(C.v2);
  cudaFree(C.perm); cudaFree(C.actA); cudaFree(C.actB);
  cudaFree(C.gate); cudaFree(C.gateB); cudaFree(C.gr);
  cudaFree(C.featCount); cudaFree(C.featStart);
  cudaFree(C.featSlotSample); cudaFree(C.featSlotSide);
  memset(&C, 0, sizeof(C));
}

} // extern "C"
