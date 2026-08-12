// eval_classical, ported from chess-cli/c_port/eval.c:19-197.
//
// This is THE ANCHOR. The engine's -DUSE_NNUE eval is
//
//	evaluate(pos) = (eval_classical(pos) + nnue_eval(pos)) / drawish_divisor(pos)
//
// and the trainer fits the net to the residue on top of eval_classical, frozen
// (chess-cli/c_port/tools/trainer.c:19-28). So this function must agree with
// the C EXACTLY. Not approximately: a constant that drifts by one centipawn
// biases every target in the corpus, and the only symptom is a weaker net.
// anchor_parity.ps1 is what proves the agreement; run it before every --pack.
//
// NOT ported, deliberately: evaluate() and drawish_divisor() (eval.c:199-278).
// The trainer wants the bare anchor -- trainer.c:183 calls eval_classical, not
// evaluate -- and the divisor is applied by the engine AFTER the residue is
// added, so folding it in here would apply it twice.

package chess

// Positional weights (eval.c:19-55; centipawns, texel-tuned by tools/texel.c).
// Kept as vars, not consts, to mirror the C's mutable globals that texel.c
// writes -- and as a reminder that these travel with pesto.go on a retune.
var (
	mobMG = [6]int{0, 4, 5, 2, 1, 0} // P N B R Q K
	mobEG = [6]int{0, 4, 5, 4, 10, 0}

	bishopPairMG = 22
	bishopPairEG = 54

	// Chess960 development terms. MINOR_HOME is POSITIVE on purpose: the PSTs
	// were tuned without these terms and already over-penalize a back-rank
	// minor, so the optimum is a small rebate (eval.c:34-40).
	minorHomeMG           = 4
	minorHomeEG           = 4
	cornerBishopBlockedMG = -19

	rookOpenMG     = 58
	rookOpenEG     = -4
	rookSemiopenMG = 20
	rookSemiopenEG = 14
	doubledMG      = -6
	doubledEG      = -22
	isolatedMG     = -14
	isolatedEG     = -10

	// Passed-pawn bonus by the pawn's relative rank (0 = own back rank).
	passedMG = [8]int{0, -2, -6, -14, 14, 14, 56, 0}
	passedEG = [8]int{0, 6, 12, 38, 63, 124, 100, 0}

	// King-danger weight per enemy attack landing on the king ring, by
	// attacker type; middlegame only.
	kingAttW        = [6]int{0, 2, 4, 3, 7, 0}
	kingDangerScale = 3
)

// pieceAttacks is eval.c:57-65. Pawns and kings return 0 -- mobility and king
// danger are only counted for N/B/R/Q.
func pieceAttacks(p, sq int, occ uint64) uint64 {
	switch p {
	case Knight:
		return knightAttacksTbl[sq]
	case Bishop:
		return bishopAttacks(sq, occ)
	case Rook:
		return rookAttacks(sq, occ)
	case Queen:
		return queenAttacks(sq, occ)
	default:
		return 0
	}
}

// EvalClassical returns the tapered hand-crafted score in centipawns from the
// SIDE TO MOVE's perspective (eval.c:192-196), which is the same perspective
// the net's output and the training target use.
func (p *Position) EvalClassical() int {
	var mg, eg [2]int
	phase := 0
	occ := p.OccAll

	// Material + piece-square tables (PeSTO).
	for side := 0; side < 2; side++ {
		for pc := 0; pc < 6; pc++ {
			bb := p.Pieces[side][pc]
			for bb != 0 {
				sq := popLSB(&bb)
				idx := sq // tables are a8 = 0
				if side == White {
					idx = sq ^ 56
				}
				mg[side] += mgValue[pc] + mgTable[pc][idx]
				eg[side] += egValue[pc] + egTable[pc][idx]
				phase += phaseInc[pc]
			}
		}
	}

	// Positional terms, per side.
	for side := 0; side < 2; side++ {
		them := side ^ 1
		own := p.Occ[side]
		ownPawns := p.Pieces[side][Pawn]
		enemyPawns := p.Pieces[them][Pawn]

		// Mobility (N/B/R/Q).
		for pc := Knight; pc <= Queen; pc++ {
			bb := p.Pieces[side][pc]
			for bb != 0 {
				sq := popLSB(&bb)
				m := popcount(pieceAttacks(pc, sq, occ) &^ own)
				mg[side] += m * mobMG[pc]
				eg[side] += m * mobEG[pc]
			}
		}

		// Bishop pair.
		if popcount(p.Pieces[side][Bishop]) >= 2 {
			mg[side] += bishopPairMG
			eg[side] += bishopPairEG
		}

		// Chess960 development (eval.c:110-129).
		{
			back := uint64(0xFF)
			if side == Black {
				back = uint64(0xFF) << 56
			}
			home := popcount((p.Pieces[side][Knight] | p.Pieces[side][Bishop]) & back)
			mg[side] += home * minorHomeMG
			eg[side] += home * minorHomeEG
			bish := p.Pieces[side][Bishop]
			if side == White {
				if bish&(uint64(1)<<0) != 0 && ownPawns&(uint64(1)<<9) != 0 {
					mg[side] += cornerBishopBlockedMG // Ba1 behind b2
				}
				if bish&(uint64(1)<<7) != 0 && ownPawns&(uint64(1)<<14) != 0 {
					mg[side] += cornerBishopBlockedMG // Bh1 behind g2
				}
			} else {
				if bish&(uint64(1)<<56) != 0 && ownPawns&(uint64(1)<<49) != 0 {
					mg[side] += cornerBishopBlockedMG // Ba8 behind b7
				}
				if bish&(uint64(1)<<63) != 0 && ownPawns&(uint64(1)<<54) != 0 {
					mg[side] += cornerBishopBlockedMG // Bh8 behind g7
				}
			}
		}

		// Rooks on open / half-open files.
		{
			bb := p.Pieces[side][Rook]
			for bb != 0 {
				sq := popLSB(&bb)
				f := fileBB[sq%8]
				if f&ownPawns == 0 {
					if f&enemyPawns == 0 {
						mg[side] += rookOpenMG
						eg[side] += rookOpenEG
					} else {
						mg[side] += rookSemiopenMG
						eg[side] += rookSemiopenEG
					}
				}
			}
		}

		// Pawn structure: passed / isolated / doubled.
		{
			bb := ownPawns
			for bb != 0 {
				sq := popLSB(&bb)
				file := sq % 8
				relRank := sq / 8
				if side == Black {
					relRank = 7 - sq/8
				}
				if passedPawnMask[side][sq]&enemyPawns == 0 {
					mg[side] += passedMG[relRank]
					eg[side] += passedEG[relRank]
				}
				if adjacentFilesBB[file]&ownPawns == 0 {
					mg[side] += isolatedMG
					eg[side] += isolatedEG
				}
			}
			// Doubled: penalize each EXTRA pawn on a file.
			for f := 0; f < 8; f++ {
				c := popcount(fileBB[f] & ownPawns)
				if c > 1 {
					mg[side] += (c - 1) * doubledMG
					eg[side] += (c - 1) * doubledEG
				}
			}
		}

		// King safety: enemy attacks landing on our king ring (mg only).
		{
			ksq := ctz(p.Pieces[side][King])
			ring := kingAttacksTbl[ksq] | (uint64(1) << uint(ksq))
			danger := 0
			for pc := Knight; pc <= Queen; pc++ {
				bb := p.Pieces[them][pc]
				for bb != 0 {
					sq := popLSB(&bb)
					danger += popcount(pieceAttacks(pc, sq, occ)&ring) * kingAttW[pc]
				}
			}
			mg[side] -= danger * kingDangerScale
		}
	}

	us := p.Stm
	them := us ^ 1
	mgScore := mg[us] - mg[them]
	egScore := eg[us] - eg[them]
	if phase > 24 {
		phase = 24 // early promotions can exceed the start phase
	}
	// Go's integer division truncates toward zero, exactly as C's does, so
	// this ports verbatim including its behaviour on negative scores.
	return (mgScore*phase + egScore*(24-phase)) / 24
}

// AnchorCP is EvalClassical clamped to the range the corpus stores it in
// (trainer.c:180-186). The clamp exists only so a freak lopsided FEN cannot
// wrap the int16 the packed corpus uses.
func (p *Position) AnchorCP() int {
	a := p.EvalClassical()
	if a > 20000 {
		a = 20000
	}
	if a < -20000 {
		a = -20000
	}
	return a
}
