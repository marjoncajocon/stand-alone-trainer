// Package train holds the two training algorithms.
//
// online.go is a deliberate near-verbatim transcription of the reference
// trainer's loop (dama-cli/c_port/tools/trainer/trainer/nnue/train_nnue.go,
// lines 588-632): AdaGrad, BATCH SIZE 1, float64, single-threaded.
//
// It is slow and it is not the production path. It exists because it is the
// ORACLE: because both it and the reference are Go using the same math/rand and
// the same math.Exp, a run with the same seed on the same data should reproduce
// the reference's weights bit-for-bit. Without that anchor, every later MSE
// difference is ambiguous between "the new optimizer is worse" and "the rewrite
// has a bug".
//
// Do not restructure this loop for speed. Four orderings are load-bearing:
//   - d1/d2 read w2[wb+j] BEFORE that element is updated on the same iteration;
//   - B1 is updated inside the same j-loop as W2, not after it;
//   - W1 is updated in feature order, and for each feature in j order;
//   - the epoch shuffle consumes the same RNG stream that initialized the
//     weights.
package train

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"nnuetrainer/internal/corpus"
	"nnuetrainer/internal/model"
)

// OnlineOpts configures the batch-1 AdaGrad path.
type OnlineOpts struct {
	Epochs int
	LR     float64
	K      float64
	Rng    *rand.Rand // MUST be the same stream that initialized the model
	Report func(epoch int, mse float64, elapsed time.Duration)
}

// Online runs batch-1 AdaGrad for Opts.Epochs over set.Train.
func Online(m *model.Model, set *corpus.Set, o OnlineOpts) {
	h := m.H
	pool := set.Pool
	train := set.Train

	g1 := make([]float64, len(m.W1))
	gb := make([]float64, len(m.B1))
	g2 := make([]float64, len(m.W2))

	acc1 := make([]float64, h)
	acc2 := make([]float64, h)
	dacc1 := make([]float64, h)
	dacc2 := make([]float64, h)

	idx := make([]int, len(train))
	for i := range idx {
		idx[i] = i
	}

	for ep := 1; ep <= o.Epochs; ep++ {
		t0 := time.Now()
		o.Rng.Shuffle(len(idx), func(i, j int) { idx[i], idx[j] = idx[j], idx[i] })

		for _, ti := range idx {
			s := train[ti]
			net := m.Forward(pool, s, acc1, acc2)
			p := 1 / (1 + math.Exp(-o.K*(float64(s.Anchor)+net)))
			g := -2 * (s.Y - p) * p * (1 - p) * o.K * 64

			wb := m.WBase(s)
			for j := 0; j < h; j++ {
				a1, a2 := model.Clip01(acc1[j]), model.Clip01(acc2[j])
				gw2 := g * (a1 - a2)
				g2[wb+j] += gw2 * gw2

				// crelu gate: a unit that is clipped passes no gradient back.
				// Read w2 BEFORE updating it, matching the reference.
				d1, d2 := 0.0, 0.0
				if acc1[j] > 0 && acc1[j] < 1 {
					d1 = g * m.W2[wb+j]
				}
				if acc2[j] > 0 && acc2[j] < 1 {
					d2 = -g * m.W2[wb+j]
				}
				m.W2[wb+j] -= o.LR * gw2 / math.Sqrt(g2[wb+j]+1e-8)

				dacc1[j], dacc2[j] = d1, d2
				db := d1 + d2
				if db != 0 {
					gb[j] += db * db
					m.B1[j] -= o.LR * db / math.Sqrt(gb[j]+1e-8)
				}
			}

			for _, f := range pool[s.Off:s.End] {
				c1 := int(f) * h
				c2 := int(m.Mirror[f]) * h
				for j := 0; j < h; j++ {
					if d := dacc1[j]; d != 0 {
						g1[c1+j] += d * d
						m.W1[c1+j] -= o.LR * d / math.Sqrt(g1[c1+j]+1e-8)
					}
					if d := dacc2[j]; d != 0 {
						g1[c2+j] += d * d
						m.W1[c2+j] -= o.LR * d / math.Sqrt(g1[c2+j]+1e-8)
					}
				}
			}
		}

		mse := m.MSE(pool, set.Valid, o.K)
		if o.Report != nil {
			o.Report(ep, mse, time.Since(t0))
		} else {
			fmt.Printf("epoch %d: valid MSE %.6f\n", ep, mse)
		}
	}
}
