// Package model holds the network weights, the float forward pass, the
// quantization contract, and the integer reference evaluators.
//
// There are TWO kernels, because there are two engine families. Each is fixed
// and non-negotiable: it is what that engine already implements in C, and the
// trainer's job is to produce weights for it, not to pick an architecture.
//
// KernelDraughts — every checkers engine's bb_nnue_score16:
//
//	g(b)     = 64 * W2[bucket] . crelu01(B1 + sum_f W1[f])
//	f(b)     = g(b) - g(mirror(b))
//	score_X  = anchor + f(b)
//
// Antisymmetry is STRUCTURAL: one shared W1 evaluated twice, once on the board
// and once on its mirror. There is no mirror head and no orientation bias, so
// the property survives quantization exactly. The material anchor is FROZEN at
// engine values and never learned — the net only ever learns the residue.
//
// KernelChess768 — chess-cli/c_port/nnue.c:
//
//	acc      = B1 + sum_f W1[f]            f is STM-RELATIVE
//	out      = B2 + sum_j W2[j]*clip(acc[j])
//	score_cp = round(out / (QA*QB))
//	score    = anchor + score_cp           anchor = eval_classical, a full HCE
//
// Here the symmetry lives in the INPUT encoding, so there is one accumulator
// and no mirror permutation, there IS an output bias, and the quantization
// scale is QA=255 / QB=64 rather than 256. Both quantized forms are exercised
// against a freshly compiled C probe before anything ships.
package model

import (
	"math"
	"math/rand"

	"nnuetrainer/internal/corpus"
	"nnuetrainer/internal/variant"
)

// Chess quantization constants (nnue.h:34-35). They must match n2h.c exactly:
//
//	w1_q = round(w1*QA)   b1_q = round(b1*QA)
//	w2_q = round(w2*QB)   b2_q = round(b2*QA*QB)
const (
	ChessQA = 255
	ChessQB = 64
)

// Model is the float64 network used by the online trainer and by every MSE
// evaluation.
type Model struct {
	V       *variant.Variant
	H       int
	NumFeat int
	Buckets int

	// Mirror is nil under the chess kernel: its features are already
	// stm-relative, so there is no second accumulator to build.
	Mirror []uint16

	W1 []float64 // [feat*H + j]
	B1 []float64 // [j]
	W2 []float64 // [bucket*H + j]

	// B2 is the scalar output bias. Chess only — the draughts kernel has no
	// such term and leaves this zero.
	B2 float64
}

// IsChess reports which kernel this model implements.
func (m *Model) IsChess() bool { return m.V != nil && m.V.IsChess() }

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
	// B2 starts at 0 and is drawn from no RNG stream on purpose: consuming a
	// value here would shift every subsequent draw and make chess runs
	// irreproducible against a net trained before it existed. Chess is new, so
	// there is no such net yet -- but the init-order contract above says the
	// stream position is load-bearing, and this keeps it true.
	m.B2 = 0
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
// pre-anchor net output.
//
// Draughts: 64*W2.(crelu(acc1)-crelu(acc2)), acc2 being the mirrored board.
// Chess:    B2 + W2.crelu(acc1). acc2 is left untouched, there is no 64x factor
// (the chess kernel's rescale lives in the quantized path, not the float one),
// and no bucket offset.
func (m *Model) Forward(pool []uint16, s corpus.Sample, acc1, acc2 []float64) float64 {
	h := m.H
	copy(acc1, m.B1)

	if m.IsChess() {
		for _, f := range pool[s.Off:s.End] {
			c := int(f) * h
			for j := 0; j < h; j++ {
				acc1[j] += m.W1[c+j]
			}
		}
		o := m.B2
		for j := 0; j < h; j++ {
			o += m.W2[j] * Clip01(acc1[j])
		}
		return o
	}

	copy(acc2, m.B1)
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

	// Chess marks the kernel. It changes the quantization scale, the forward
	// pass and the generated header's shape, so it travels with the weights
	// rather than being re-derived at every call site.
	Chess bool

	Mirror []uint16
	W1     []int16
	B1     []int32
	W2     []int16
	B2     int32 // chess only
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
//
// The two kernels use DIFFERENT scales. Draughts is x256 throughout. Chess is
// QA=255 for the feature transformer, QB=64 for the output weights and QA*QB
// for the output bias (n2h.c:6-9) — mixing them up produces a header that
// compiles, loads and evaluates to nonsense.
func (m *Model) Quantize() *Quant {
	q := &Quant{
		H: m.H, NumFeat: m.NumFeat, Buckets: m.Buckets,
		Chess:  m.IsChess(),
		Mirror: m.Mirror,
		W1:     make([]int16, len(m.W1)),
		B1:     make([]int32, len(m.B1)),
		W2:     make([]int16, len(m.W2)),
	}
	if q.Chess {
		for i, v := range m.W1 {
			q.W1[i] = ClipI16(v * ChessQA)
		}
		for j, v := range m.B1 {
			q.B1[j] = int32(math.Round(v * ChessQA))
		}
		for j, v := range m.W2 {
			q.W2[j] = ClipI16(v * ChessQB)
		}
		q.B2 = int32(math.Round(m.B2 * ChessQA * ChessQB))
		return q
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

func clampQA(x int32) int64 {
	if x < 0 {
		return 0
	}
	if x > ChessQA {
		return ChessQA
	}
	return int64(x)
}

// ScoreCP is the chess integer reference: a literal port of nnue_eval
// (chess-cli/c_port/nnue.c:73-88), returning centipawns from the side to
// move's perspective.
//
// The accumulator is int32 ON PURPOSE, and there is no int16 variant. The C
// says so outright (position.h:34-42): "int32, not int16: pre-activation sums
// can exceed the int16 range with this net's weight magnitudes". So the
// draughts kernel's narrow-accumulator gate has no counterpart here, and
// parity_chess.ps1 reports that leg as not applicable rather than passing it.
//
// The final rescale rounds half AWAY from zero via the +/- denom/2 trick, and
// Go's integer division truncates toward zero exactly as C's does, so the
// expression ports verbatim.
func (q *Quant) ScoreCP(feats []uint16) int {
	h := q.H
	acc := make([]int32, h)
	copy(acc, q.B1)
	for _, f := range feats {
		c := int(f) * h
		for j := 0; j < h; j++ {
			acc[j] += int32(q.W1[c+j])
		}
	}
	out := int64(q.B2)
	for j := 0; j < h; j++ {
		out += clampQA(acc[j]) * int64(q.W2[j])
	}
	const denom = int64(ChessQA) * int64(ChessQB)
	if out >= 0 {
		return int((out + denom/2) / denom)
	}
	return int((out - denom/2) / denom)
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
//
// It is reported for chess too, but as INFORMATION, not a gate: that kernel's
// accumulator is int32 in the engine as well (position.h:34-42), so exceeding
// the int16 range there is expected and harmless. Only the draughts engines
// narrow to int16 and can silently overflow.
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
		var sc float64
		if q.Chess {
			// ScoreCP is already centipawns; the draughts /16 is that kernel's
			// own fixed-point convention and does not apply.
			sc = float64(q.ScoreCP(pool[s.Off:s.End]))
		} else {
			sc = float64(q.Score16(pool[s.Off:s.End], int(s.Bucket))) / 16
		}
		p := 1 / (1 + math.Exp(-k*(float64(s.Anchor)+sc)))
		d := s.Y - p
		e += d * d
	}
	return e / float64(len(set))
}
