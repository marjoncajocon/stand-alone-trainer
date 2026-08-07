// Package model holds the network weights, the float forward pass, the
// quantization contract, and the integer reference evaluator.
//
// Topology is fixed and non-negotiable, because it is what every engine's
// bb_nnue_score16 implements:
//
//	g(b)     = 64 * W2[bucket] . crelu01(B1 + sum_f W1[f])
//	f(b)     = g(b) - g(mirror(b))
//	score_X  = anchor + f(b)
//
// Antisymmetry is STRUCTURAL: one shared W1 evaluated twice, once on the board
// and once on its mirror. There is no mirror head and no orientation bias, so
// the property survives quantization exactly. The material anchor is FROZEN at
// engine values and never learned — the net only ever learns the residue.
package model

import (
	"math"
	"math/rand"

	"nnuetrainer/internal/corpus"
	"nnuetrainer/internal/variant"
)

// Model is the float64 network used by the online trainer and by every MSE
// evaluation.
type Model struct {
	V       *variant.Variant
	H       int
	NumFeat int
	Buckets int
	Mirror  []uint16

	W1 []float64 // [feat*H + j]
	B1 []float64 // [j]
	W2 []float64 // [bucket*H + j]
}

// New allocates and initializes a model.
//
// The caller passes the RNG rather than a seed because the reference trainer
// uses ONE stream for both weight init and every epoch's shuffle. Creating a
// second RNG here would give identical starting weights but a different
// shuffle order, and the resulting net would not reproduce.
//
// The init order (all of W1, then B1, then W2) is equally load-bearing: it is
// what fixes how much of the stream init consumes.
func New(v *variant.Variant, h, buckets int, rng *rand.Rand) *Model {
	m := &Model{
		V: v, H: h, NumFeat: v.NumFeat, Buckets: buckets,
		Mirror: v.Mirror(),
		W1:     make([]float64, v.NumFeat*h),
		B1:     make([]float64, h),
		W2:     make([]float64, buckets*h),
	}
	for i := range m.W1 {
		m.W1[i] = (rng.Float64()*2 - 1) * 0.3
	}
	for j := range m.B1 {
		m.B1[j] = 0.05
	}
	for j := range m.W2 {
		m.W2[j] = (rng.Float64()*2 - 1) * 0.5
	}
	return m
}

// WBase is the offset of a sample's output row; folds to 0 for a single-W2 net.
func (m *Model) WBase(s corpus.Sample) int {
	if m.Buckets == 1 {
		return 0
	}
	return int(s.Bucket) * m.H
}

func Clip01(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

// Forward fills acc1/acc2 (caller-owned scratch of length H) and returns the
// pre-anchor net output 64*W2.(crelu(acc1)-crelu(acc2)).
func (m *Model) Forward(pool []uint16, s corpus.Sample, acc1, acc2 []float64) float64 {
	copy(acc1, m.B1)
	copy(acc2, m.B1)
	h := m.H
	for _, f := range pool[s.Off:s.End] {
		c1 := int(f) * h
		c2 := int(m.Mirror[f]) * h
		for j := 0; j < h; j++ {
			acc1[j] += m.W1[c1+j]
			acc2[j] += m.W1[c2+j]
		}
	}
	o := 0.0
	wb := m.WBase(s)
	for j := 0; j < h; j++ {
		a1, a2 := Clip01(acc1[j]), Clip01(acc2[j])
		o += m.W2[wb+j] * (a1 - a2)
	}
	return 64 * o
}

// MSE is the held-out mean squared error of the float model.
func (m *Model) MSE(pool []uint16, set []corpus.Sample, k float64) float64 {
	if len(set) == 0 {
		return math.NaN()
	}
	acc1 := make([]float64, m.H)
	acc2 := make([]float64, m.H)
	var e float64
	for _, s := range set {
		net := m.Forward(pool, s, acc1, acc2)
		p := 1 / (1 + math.Exp(-k*(float64(s.Anchor)+net)))
		d := s.Y - p
		e += d * d
	}
	return e / float64(len(set))
}

// AnchorMSE is the baseline: what the frozen material anchor alone scores on
// the same holdout. Every net must beat this or it is worse than counting
// pieces.
func AnchorMSE(set []corpus.Sample, k float64) float64 {
	if len(set) == 0 {
		return math.NaN()
	}
	var e float64
	for _, s := range set {
		p := 1 / (1 + math.Exp(-k*float64(s.Anchor)))
		d := s.Y - p
		e += d * d
	}
	return e / float64(len(set))
}

// ── quantization ────────────────────────────────────────────────────────────
//
// The exact-parity contract with the C engine:
//
//	W1q = round(W1*256) int16   B1q = round(B1*256) int32
//	W2q = round(W2*256) int16   aq  = clamp(acc, 0, 256)
//	o = sum W2q[j]*(a1q[j]-a2q[j]),  score16 = o/64   (truncating, like C)
//
// score16 is 16x engine units; the engine call site keeps its /16.

// Quant is the quantized net — the artifact that actually plays.
type Quant struct {
	H       int
	NumFeat int
	Buckets int
	Mirror  []uint16
	W1      []int16
	B1      []int32
	W2      []int16
}

func ClipI16(x float64) int16 {
	r := math.Round(x)
	if r > 32000 {
		return 32000
	}
	if r < -32000 {
		return -32000
	}
	return int16(r)
}

// Quantize converts the float model to its shipping form.
func (m *Model) Quantize() *Quant {
	q := &Quant{
		H: m.H, NumFeat: m.NumFeat, Buckets: m.Buckets, Mirror: m.Mirror,
		W1: make([]int16, len(m.W1)),
		B1: make([]int32, len(m.B1)),
		W2: make([]int16, len(m.W2)),
	}
	for i, v := range m.W1 {
		q.W1[i] = ClipI16(v * 256)
	}
	for j, v := range m.B1 {
		q.B1[j] = int32(math.Round(v * 256))
	}
	for j, v := range m.W2 {
		q.W2[j] = ClipI16(v * 256)
	}
	return q
}

func clampQ(x int32) int {
	if x < 0 {
		return 0
	}
	if x > 256 {
		return 256
	}
	return int(x)
}

// Score16 is the integer reference the C engine must match bit-exactly. It uses
// an int32 accumulator, matching probe_nnue.c.
func (q *Quant) Score16(feats []uint16, bucket int) int {
	h := q.H
	acc1 := make([]int32, h)
	acc2 := make([]int32, h)
	copy(acc1, q.B1)
	copy(acc2, q.B1)
	for _, f := range feats {
		c1 := int(f) * h
		c2 := int(q.Mirror[f]) * h
		for j := 0; j < h; j++ {
			acc1[j] += int32(q.W1[c1+j])
			acc2[j] += int32(q.W1[c2+j])
		}
	}
	wb := 0
	if q.Buckets > 1 {
		wb = bucket * h
	}
	o := 0
	for j := 0; j < h; j++ {
		a1, a2 := clampQ(acc1[j]), clampQ(acc2[j])
		o += int(q.W2[wb+j]) * (a1 - a2)
	}
	return o / 64
}

// Score16Narrow reproduces the SHIPPED kernel exactly: search.c accumulates
// into int16 and seeds it by narrowing the int32 NN_B1. That is only legal
// while the pre-clamp sum stays inside +/-32767 — the clamp to [0,256] happens
// AFTER the sum, so an overflow silently changes the eval with no symptom
// except worse play.
//
// Score16 and Score16Narrow agreeing on real boards is a TEST of that safety
// property, complementing probe_accbound's proof of it.
func (q *Quant) Score16Narrow(feats []uint16, bucket int) int {
	h := q.H
	acc1 := make([]int16, h)
	acc2 := make([]int16, h)
	for j := 0; j < h; j++ {
		acc1[j] = int16(q.B1[j])
		acc2[j] = int16(q.B1[j])
	}
	for _, f := range feats {
		c1 := int(f) * h
		c2 := int(q.Mirror[f]) * h
		for j := 0; j < h; j++ {
			acc1[j] += q.W1[c1+j]
			acc2[j] += q.W1[c2+j]
		}
	}
	wb := 0
	if q.Buckets > 1 {
		wb = bucket * h
	}
	o := 0
	for j := 0; j < h; j++ {
		a1, a2 := clampQ(int32(acc1[j])), clampQ(int32(acc2[j]))
		o += int(q.W2[wb+j]) * (a1 - a2)
	}
	return o / 64
}

// AccBound measures the worst-case pre-clamp accumulator reached over a sample
// set. This is the same check probe_accbound.c performs, so a retrain can be
// cleared without leaving the trainer.
func (q *Quant) AccBound(pool []uint16, sets ...[]corpus.Sample) (lo, hi int32) {
	h := q.H
	acc := make([]int32, h)
	for _, set := range sets {
		for _, s := range set {
			copy(acc, q.B1)
			for _, f := range pool[s.Off:s.End] {
				c := int(f) * h
				for j := 0; j < h; j++ {
					acc[j] += int32(q.W1[c+j])
				}
			}
			for j := 0; j < h; j++ {
				if acc[j] > hi {
					hi = acc[j]
				}
				if acc[j] < lo {
					lo = acc[j]
				}
			}
		}
	}
	return
}

// MSE is the held-out error of the QUANTIZED net — the decisive metric, since
// it measures the artifact that actually plays rather than the float model.
func (q *Quant) MSE(pool []uint16, set []corpus.Sample, k float64) float64 {
	if len(set) == 0 {
		return math.NaN()
	}
	var e float64
	for _, s := range set {
		sc := float64(q.Score16(pool[s.Off:s.End], int(s.Bucket))) / 16
		p := 1 / (1 + math.Exp(-k*(float64(s.Anchor)+sc)))
		d := s.Y - p
		e += d * d
	}
	return e / float64(len(set))
}
