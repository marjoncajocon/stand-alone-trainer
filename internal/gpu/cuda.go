//go:build cuda

package gpu

/*
#cgo LDFLAGS: -lcudart -lstdc++
#include <stdlib.h>
int  nt_gpu_available(void);
int  nt_gpu_name(int ordinal, char *out, int cap);
int  nt_gpu_upload(int ordinal, int H, int numFeat, int buckets, int batch,
                   float k, int isAdam, const unsigned short *pool, int poolLen,
                   const int *off, const int *end, const short *anchor,
                   const unsigned char *bucket, const float *y, int nTotal,
                   int nTrain, int nValid, const unsigned short *mirror,
                   const float *w1, const float *b1, const float *w2);
int  nt_gpu_step(const int *perm, int n, const int *slotSample,
                 const int *slotSide, const int *featCount, const int *featStart,
                 int nSlots, float lr, float bc1, float bc2);
int  nt_gpu_valid_mse(double *out);
int  nt_gpu_download(float *w1, float *b1, float *w2);
void nt_gpu_close(void);
const char *nt_gpu_error(void);
*/
import "C"

import (
	"fmt"
	"math"
	"unsafe"
)

// Available reports whether a CUDA device is present.
func Available() bool { return C.nt_gpu_available() > 0 }

// Backend names the compiled-in backend.
func Backend() string { return "CUDA (cgo)" }

type cudaDev struct {
	name    string
	p       *Params
	c       *Corpus
	nTrain  int
	nValid  int
	slotS   []C.int
	slotSid []C.int
	fCount  []C.int
	fStart  []C.int
	step    int64
}

func lastErr() error { return fmt.Errorf("cuda: %s", C.GoString(C.nt_gpu_error())) }

// Open selects a CUDA device.
func Open(ordinal int) (Device, error) {
	if C.nt_gpu_available() <= 0 {
		return nil, fmt.Errorf("no CUDA device visible (nvidia-smi -L to check)")
	}
	buf := make([]byte, 256)
	if C.nt_gpu_name(C.int(ordinal), (*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf))) != 0 {
		return nil, lastErr()
	}
	n := 0
	for n < len(buf) && buf[n] != 0 {
		n++
	}
	return &cudaDev{name: string(buf[:n])}, nil
}

func (d *cudaDev) Name() string { return d.name }

func (d *cudaDev) Upload(c *Corpus, w *Weights, p *Params) error {
	d.p, d.c = p, c
	nTotal := len(c.Off)
	d.nTrain, d.nValid = c.TrainCount, c.ValidCount

	isAdam := 0
	if p.Opt != "adagrad" {
		isAdam = 1
	}
	rc := C.nt_gpu_upload(0, C.int(p.H), C.int(p.NumFeat), C.int(p.Buckets),
		C.int(p.Batch), C.float(p.K), C.int(isAdam),
		(*C.ushort)(unsafe.Pointer(&c.Pool[0])), C.int(len(c.Pool)),
		(*C.int)(unsafe.Pointer(&c.Off[0])), (*C.int)(unsafe.Pointer(&c.End[0])),
		(*C.short)(unsafe.Pointer(&c.Anchor[0])),
		(*C.uchar)(unsafe.Pointer(&c.Bucket[0])),
		(*C.float)(unsafe.Pointer(&c.Y[0])), C.int(nTotal),
		C.int(d.nTrain), C.int(d.nValid),
		(*C.ushort)(unsafe.Pointer(&p.Mirror[0])),
		(*C.float)(unsafe.Pointer(&w.W1[0])),
		(*C.float)(unsafe.Pointer(&w.B1[0])),
		(*C.float)(unsafe.Pointer(&w.W2[0])))
	if rc != 0 {
		return lastErr()
	}
	d.slotS = make([]C.int, p.Batch*128)
	d.slotSid = make([]C.int, p.Batch*128)
	d.fCount = make([]C.int, p.NumFeat)
	d.fStart = make([]C.int, p.NumFeat)
	return nil
}

// RunEpoch consumes perm in batches, building the inverted index per step on
// the host (a counting sort over <= NumFeat buckets — microseconds, and
// trivially deterministic).
func (d *cudaDev) RunEpoch(perm []int32, lr float32) error {
	p := d.p
	c := d.c
	batch := p.Batch
	steps := len(perm) / batch
	permBuf := make([]C.int, batch)

	for st := 0; st < steps; st++ {
		ids := perm[st*batch : st*batch+batch]
		for i, id := range ids {
			permBuf[i] = C.int(id)
		}
		for i := range d.fCount {
			d.fCount[i] = 0
		}
		// pass 1: count
		nSlots := 0
		for _, id := range ids {
			for t := c.Off[id]; t < c.End[id]; t++ {
				f := c.Pool[t]
				d.fCount[f]++
				d.fCount[p.Mirror[f]]++
				nSlots += 2
			}
		}
		if nSlots > len(d.slotS) {
			d.slotS = make([]C.int, nSlots)
			d.slotSid = make([]C.int, nSlots)
		}
		// prefix sums
		run := C.int(0)
		for i := range d.fStart {
			d.fStart[i] = run
			run += d.fCount[i]
		}
		// pass 2: scatter, in ascending sample order so the reduction order is
		// fixed and matches the CPU chunk walk.
		cur := make([]C.int, p.NumFeat)
		copy(cur, d.fStart)
		for s, id := range ids {
			for t := c.Off[id]; t < c.End[id]; t++ {
				f := c.Pool[t]
				m := p.Mirror[f]
				d.slotS[cur[f]] = C.int(s)
				d.slotSid[cur[f]] = 0
				cur[f]++
				d.slotS[cur[m]] = C.int(s)
				d.slotSid[cur[m]] = 1
				cur[m]++
			}
		}
		// Bias correction in float64, identical to internal/train/batch.go's
		// apply(). Computing it here rather than in the kernel is what keeps
		// the CPU and GPU paths bit-comparable.
		d.step++
		t := float64(d.step)
		bc1 := float32(1 - math.Pow(0.9, t))
		bc2 := float32(1 - math.Pow(0.999, t))

		rc := C.nt_gpu_step(&permBuf[0], C.int(batch), &d.slotS[0], &d.slotSid[0],
			&d.fCount[0], &d.fStart[0], C.int(nSlots), C.float(lr),
			C.float(bc1), C.float(bc2))
		if rc != 0 {
			return lastErr()
		}
	}
	return nil
}

func (d *cudaDev) ValidMSE(_ *Corpus) (float64, error) {
	var out C.double
	if C.nt_gpu_valid_mse(&out) != 0 {
		return 0, lastErr()
	}
	return float64(out), nil
}

func (d *cudaDev) Download(w *Weights) error {
	if C.nt_gpu_download((*C.float)(unsafe.Pointer(&w.W1[0])),
		(*C.float)(unsafe.Pointer(&w.B1[0])),
		(*C.float)(unsafe.Pointer(&w.W2[0]))) != 0 {
		return lastErr()
	}
	return nil
}

func (d *cudaDev) Close() error { C.nt_gpu_close(); return nil }
