package train

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"time"

	"nnuetrainer/internal/corpus"
	"nnuetrainer/internal/model"
)

// nChunks is the FIXED number of gradient partitions per step.
//
// This is what makes --threads a pure performance knob. Each chunk owns a
// contiguous slice of the batch and its own gradient buffer; the reduction then
// walks chunks 0..nChunks-1 in index order. Float addition is not associative,
// so the ORDER must be fixed - but it does not have to depend on how many
// goroutines happen to be running. Partitioning per worker instead would make
// every result a function of --threads, which is exactly the kind of silent
// irreproducibility this project has been bitten by before.
const nChunks = 16

// BatchOpts configures the minibatch path.
type BatchOpts struct {
	Epochs       int
	Batch        int
	LR           float32
	K            float32
	Opt          string // "adam" or "adagrad"
	Schedule     string // "const" or "cosine"
	WarmupEpochs int
	Threads      int
	Rng          *rand.Rand
	Report       func(epoch int, mse float64, elapsed time.Duration, lr float32)
}

// Batch is the float32 minibatch trainer.
//
// float32 everywhere is deliberate: it is what a CUDA kernel computes in, so a
// CPU run and a GPU run of the same steps can be compared byte-for-byte rather
// than "within tolerance". Master weights are float32 too - keeping float64
// masters would make the CPU path unreproducible on the GPU for no measurable
// accuracy gain at this network size.
type Batch struct {
	H, NumFeat, Buckets int
	Mirror              []uint16

	W1, B1, W2 []float32

	// optimizer state
	m1, v1 []float32 // W1
	mb, vb []float32 // B1
	m2, v2 []float32 // W2
	step   int64
}

// NewBatch snapshots a float64 model into float32 training form.
func NewBatch(m *model.Model) *Batch {
	b := &Batch{
		H: m.H, NumFeat: m.NumFeat, Buckets: m.Buckets, Mirror: m.Mirror,
		W1: f32(m.W1), B1: f32(m.B1), W2: f32(m.W2),
	}
	b.m1 = make([]float32, len(b.W1))
	b.v1 = make([]float32, len(b.W1))
	b.mb = make([]float32, len(b.B1))
	b.vb = make([]float32, len(b.B1))
	b.m2 = make([]float32, len(b.W2))
	b.v2 = make([]float32, len(b.W2))
	return b
}

// Into writes the trained float32 weights back into the float64 model, which is
// what gets quantized and emitted.
func (b *Batch) Into(m *model.Model) {
	for i, v := range b.W1 {
		m.W1[i] = float64(v)
	}
	for i, v := range b.B1 {
		m.B1[i] = float64(v)
	}
	for i, v := range b.W2 {
		m.W2[i] = float64(v)
	}
}

func f32(x []float64) []float32 {
	o := make([]float32, len(x))
	for i, v := range x {
		o[i] = float32(v)
	}
	return o
}

func clip01f(x float32) float32 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

// grads is one chunk's gradient accumulator.
type grads struct {
	w1, b1, w2 []float32
	acc1, acc2 []float32
	d1, d2     []float32
}

func newGrads(b *Batch) *grads {
	return &grads{
		w1: make([]float32, len(b.W1)), b1: make([]float32, len(b.B1)),
		w2:   make([]float32, len(b.W2)),
		acc1: make([]float32, b.H), acc2: make([]float32, b.H),
		d1: make([]float32, b.H), d2: make([]float32, b.H),
	}
}

func (g *grads) zero() {
	for i := range g.w1 {
		g.w1[i] = 0
	}
	for i := range g.b1 {
		g.b1[i] = 0
	}
	for i := range g.w2 {
		g.w2[i] = 0
	}
}

// accumulate adds the gradients of one contiguous slice of the batch.
//
// This is the exact computation the CUDA kernel must reproduce. Written out so
// the two can be diffed line by line:
//
//	acc1 = B1 + sum_f W1[f]          acc2 = B1 + sum_f W1[mirror(f)]
//	net  = 64 * sum_j W2[wb+j]*(crelu(acc1[j]) - crelu(acc2[j]))
//	p    = sigmoid(K*(anchor + net))
//	gr   = -2*(y-p)*p*(1-p)*K*64
//	dW2[wb+j] += gr*(crelu(acc1[j]) - crelu(acc2[j]))
//	d1[j] = (0<acc1[j]<1) ?  gr*W2[wb+j] : 0
//	d2[j] = (0<acc2[j]<1) ? -gr*W2[wb+j] : 0
//	dB1[j] += d1[j] + d2[j]
//	dW1[f*H+j] += d1[j];  dW1[mirror(f)*H+j] += d2[j]
func (b *Batch) accumulate(g *grads, pool []uint16, batch []corpus.Sample, k float32) {
	h := b.H
	for _, s := range batch {
		copy(g.acc1, b.B1)
		copy(g.acc2, b.B1)
		for _, f := range pool[s.Off:s.End] {
			c1 := int(f) * h
			c2 := int(b.Mirror[f]) * h
			for j := 0; j < h; j++ {
				g.acc1[j] += b.W1[c1+j]
				g.acc2[j] += b.W1[c2+j]
			}
		}
		wb := 0
		if b.Buckets > 1 {
			wb = int(s.Bucket) * h
		}
		var o float32
		for j := 0; j < h; j++ {
			o += b.W2[wb+j] * (clip01f(g.acc1[j]) - clip01f(g.acc2[j]))
		}
		net := 64 * o
		p := sigmoidf(k * (float32(s.Anchor) + net))
		gr := -2 * (float32(s.Y) - p) * p * (1 - p) * k * 64

		for j := 0; j < h; j++ {
			a1, a2 := clip01f(g.acc1[j]), clip01f(g.acc2[j])
			g.w2[wb+j] += gr * (a1 - a2)
			var d1, d2 float32
			if g.acc1[j] > 0 && g.acc1[j] < 1 {
				d1 = gr * b.W2[wb+j]
			}
			if g.acc2[j] > 0 && g.acc2[j] < 1 {
				d2 = -gr * b.W2[wb+j]
			}
			g.d1[j], g.d2[j] = d1, d2
			g.b1[j] += d1 + d2
		}
		for _, f := range pool[s.Off:s.End] {
			c1 := int(f) * h
			c2 := int(b.Mirror[f]) * h
			for j := 0; j < h; j++ {
				g.w1[c1+j] += g.d1[j]
				g.w1[c2+j] += g.d2[j]
			}
		}
	}
}

func (b *Batch) apply(g *grads, lr, scale float32, opt string) {
	b.step++
	switch opt {
	case "adagrad":
		adagrad(b.W1, g.w1, b.v1, lr, scale)
		adagrad(b.B1, g.b1, b.vb, lr, scale)
		adagrad(b.W2, g.w2, b.v2, lr, scale)
	default: // adam
		t := float64(b.step)
		bc1 := float32(1 - math.Pow(0.9, t))
		bc2 := float32(1 - math.Pow(0.999, t))
		adam(b.W1, g.w1, b.m1, b.v1, lr, scale, bc1, bc2)
		adam(b.B1, g.b1, b.mb, b.vb, lr, scale, bc1, bc2)
		adam(b.W2, g.w2, b.m2, b.v2, lr, scale, bc1, bc2)
	}
}

func adagrad(w, g, acc []float32, lr, scale float32) {
	for i := range w {
		gi := g[i] * scale
		acc[i] += gi * gi
		w[i] -= lr * gi / float32(math.Sqrt(float64(acc[i])+1e-8))
	}
}

func adam(w, g, m, v []float32, lr, scale, bc1, bc2 float32) {
	for i := range w {
		gi := g[i] * scale
		m[i] = 0.9*m[i] + 0.1*gi
		v[i] = 0.999*v[i] + 0.001*gi*gi
		mh := m[i] / bc1
		vh := v[i] / bc2
		w[i] -= lr * mh / (float32(math.Sqrt(float64(vh))) + 1e-8)
	}
}

// Run trains for o.Epochs. Returns after writing nothing - the caller copies
// weights back with Into.
func (b *Batch) Run(set *corpus.Set, o BatchOpts) {
	if o.Batch <= 0 {
		o.Batch = 8192
	}
	if o.Threads <= 0 {
		o.Threads = runtime.NumCPU()
	}
	pool := set.Pool
	train := set.Train

	chunks := make([]*grads, nChunks)
	for i := range chunks {
		chunks[i] = newGrads(b)
	}

	idx := make([]int32, len(train))
	for i := range idx {
		idx[i] = int32(i)
	}
	buf := make([]corpus.Sample, o.Batch)

	stepsPerEpoch := len(train) / o.Batch
	if stepsPerEpoch < 1 {
		stepsPerEpoch = 1
	}

	for ep := 1; ep <= o.Epochs; ep++ {
		t0 := time.Now()
		lr := ScheduleLR(o, ep)

		o.Rng.Shuffle(len(idx), func(i, j int) { idx[i], idx[j] = idx[j], idx[i] })

		for st := 0; st < stepsPerEpoch; st++ {
			base := st * o.Batch
			n := o.Batch
			if base+n > len(train) {
				n = len(train) - base
			}
			if n <= 0 {
				break
			}
			for i := 0; i < n; i++ {
				buf[i] = train[idx[base+i]]
			}
			bs := buf[:n]

			// Fixed chunk boundaries -> fixed reduction order.
			bounds := make([][2]int, nChunks)
			for c := 0; c < nChunks; c++ {
				lo := c * n / nChunks
				hi := (c + 1) * n / nChunks
				bounds[c] = [2]int{lo, hi}
			}

			var wg sync.WaitGroup
			sem := make(chan struct{}, o.Threads)
			for c := 0; c < nChunks; c++ {
				wg.Add(1)
				go func(c int) {
					defer wg.Done()
					sem <- struct{}{}
					defer func() { <-sem }()
					g := chunks[c]
					g.zero()
					lo, hi := bounds[c][0], bounds[c][1]
					if lo < hi {
						b.accumulate(g, pool, bs[lo:hi], o.K)
					}
				}(c)
			}
			wg.Wait()

			// Deterministic reduce: chunk order, always.
			tot := chunks[0]
			for c := 1; c < nChunks; c++ {
				g := chunks[c]
				for i := range tot.w1 {
					tot.w1[i] += g.w1[i]
				}
				for i := range tot.b1 {
					tot.b1[i] += g.b1[i]
				}
				for i := range tot.w2 {
					tot.w2[i] += g.w2[i]
				}
			}
			b.apply(tot, lr, 1/float32(n), o.Opt)
		}

		mse := b.MSE(pool, set.Valid, o.K)
		if o.Report != nil {
			o.Report(ep, mse, time.Since(t0), lr)
		} else {
			fmt.Printf("epoch %d: valid MSE %.6f (lr %.2e)\n", ep, mse, lr)
		}
	}
}

// ScheduleLR is exported so the GPU driver in cmd/trainer uses the identical
// schedule rather than a second copy that can drift.
func ScheduleLR(o BatchOpts, ep int) float32 {
	if ep <= o.WarmupEpochs && o.WarmupEpochs > 0 {
		return o.LR * float32(ep) / float32(o.WarmupEpochs+1)
	}
	if o.Schedule != "cosine" {
		return o.LR
	}
	span := o.Epochs - o.WarmupEpochs
	if span <= 0 {
		return o.LR
	}
	// t is computed off ep-1 so the FIRST post-warmup epoch gets the full LR and
	// the LAST still gets a small non-zero one. Using ep directly puts t=1 on the
	// final epoch, i.e. lr=0, which silently wastes it.
	t := float64(ep-1-o.WarmupEpochs) / float64(span)
	return o.LR * float32(0.5*(1+math.Cos(math.Pi*t)))
}

// MSE evaluates the float32 model on the holdout.
func (b *Batch) MSE(pool []uint16, set []corpus.Sample, k float32) float64 {
	if len(set) == 0 {
		return math.NaN()
	}
	h := b.H
	acc1 := make([]float32, h)
	acc2 := make([]float32, h)
	var e float64
	for _, s := range set {
		copy(acc1, b.B1)
		copy(acc2, b.B1)
		for _, f := range pool[s.Off:s.End] {
			c1 := int(f) * h
			c2 := int(b.Mirror[f]) * h
			for j := 0; j < h; j++ {
				acc1[j] += b.W1[c1+j]
				acc2[j] += b.W1[c2+j]
			}
		}
		wb := 0
		if b.Buckets > 1 {
			wb = int(s.Bucket) * h
		}
		var o float32
		for j := 0; j < h; j++ {
			o += b.W2[wb+j] * (clip01f(acc1[j]) - clip01f(acc2[j]))
		}
		p := sigmoidf(k * (float32(s.Anchor) + 64*o))
		d := s.Y - float64(p)
		e += d * d
	}
	return e / float64(len(set))
}

// ── float32 sigmoid ─────────────────────────────────────────────────────────
//
// Hand-written in explicit float32 ops with no libm call, so the identical
// source can be compiled by nvcc (with -fmad=false) and produce the same bits.
// Using math.Exp on a float64 here would make CPU and GPU results incomparable
// and reduce the GPU acceptance gate from "byte-identical" to "close enough".

func sigmoidf(x float32) float32 { return 1 / (1 + expf(-x)) }

func expf(x float32) float32 {
	if x >= 88.7 {
		return float32(math.Inf(1))
	}
	if x <= -87.4 {
		return 0
	}
	const log2e = float32(1.4426950408889634)
	const ln2hi = float32(0.693359375)     // exactly representable
	const ln2lo = float32(-2.1219444e-4)   // remainder of ln2
	kf := x * log2e
	var k int32
	if kf >= 0 {
		k = int32(kf + 0.5)
	} else {
		k = int32(kf - 0.5)
	}
	fk := float32(k)
	r := (x - fk*ln2hi) - fk*ln2lo
	// degree-6 Taylor on |r| <= ln2/2; truncation error ~ r^7/5040 < 1 ulp.
	p := float32(1.0 / 720)
	p = p*r + float32(1.0/120)
	p = p*r + float32(1.0/24)
	p = p*r + float32(1.0/6)
	p = p*r + float32(0.5)
	p = p*r + 1
	p = p*r + 1
	return ldexpf(p, k)
}

func ldexpf(x float32, k int32) float32 {
	if k > 127 {
		x *= math.Float32frombits(uint32(127+127) << 23)
		k -= 127
		if k > 127 {
			k = 127
		}
	} else if k < -126 {
		x *= math.Float32frombits(uint32(-126+127) << 23)
		k += 126
		if k < -126 {
			k = -126
		}
	}
	return x * math.Float32frombits(uint32(k+127)<<23)
}
