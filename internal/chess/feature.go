// NNUE input features, ported from chess-cli/c_port/nnue.h:32-42 and the
// trainer's matching extractor (tools/trainer.c:83-97).
//
// 768 inputs = rel_color(2) x piece_type(6) x square(64), built from the SIDE
// TO MOVE's perspective: rel_color 0 is the mover's own pieces, and when the
// mover is Black every square is rank-flipped (sq^56). That makes the eval
// color-symmetric BY CONSTRUCTION -- it is why `chess evalsym` passes and why
// the net's output is already stm-relative centipawns.
//
// This is the structural difference from the six draughts variants. They use
// ABSOLUTE features and recover antisymmetry at the OUTPUT, as
// f(b) = g(b) - g(mirror(b)). Chess puts the symmetry in the INPUT instead, so
// it has one accumulator and no mirror permutation. Do not try to unify them.

package chess

// NumFeatures is NNUE_IN (nnue.h:32).
const NumFeatures = 768

// MaxFeatures is the most active features a legal position can have: 32 pieces
// (trainer.c:55 sizes its per-sample count as <= 32).
const MaxFeatures = 32

// featureIndex is nnue_feature (nnue.h:40-42): relColor*384 + pt*64 + rsq,
// where rsq is already flipped into the stm's view.
func featureIndex(relColor, pt, rsq int) int {
	return relColor*384 + pt*64 + rsq
}

// AppendFeatures appends this position's active feature indices to dst and
// returns the extended slice. dst is a shared pool, so nothing is allocated
// per position.
//
// Iteration order is by color then piece type then ascending square, matching
// nnue_refresh (nnue.c:34-42). Order does not change the accumulator -- it is
// a sum -- but it is kept identical so a feature list dumped from either side
// can be diffed directly.
func (p *Position) AppendFeatures(dst []uint16) []uint16 {
	stm := p.Stm
	for c := 0; c < 2; c++ {
		relColor := 1
		if c == stm {
			relColor = 0
		}
		for pt := Pawn; pt <= King; pt++ {
			bb := p.Pieces[c][pt]
			for bb != 0 {
				sq := popLSB(&bb)
				rsq := sq
				if stm == Black {
					rsq = sq ^ 56
				}
				dst = append(dst, uint16(featureIndex(relColor, pt, rsq)))
			}
		}
	}
	return dst
}

// PieceCount is the total number of pieces on the board. Chess has a single W2
// row so this does not select a bucket; it is here for corpus statistics and
// for the sanity check that a position never exceeds MaxFeatures.
func (p *Position) PieceCount() int {
	return popcount(p.OccAll)
}
