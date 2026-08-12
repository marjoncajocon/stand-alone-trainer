// Package chess is a port of the parts of chess-cli/c_port the trainer needs:
// the board representation, FEN parsing, and eval_classical -- the frozen
// anchor the NNUE residue is fitted against.
//
// It is a PORT, not a reimplementation. Every function here mirrors a specific
// C function and is gated against it (anchor_parity.ps1). Nothing may be
// "improved" while porting: an eval that is better but different fits the
// residue against an anchor the engine does not compute, which has no symptom
// except a net that plays worse.
//
// Deliberately ABSENT, because eval_classical never reads them: movegen,
// make/unmake, zobrist keys, and the between_bb / line_bb / pawn_attacks_tbl
// tables that exist only to serve them (bitboard.c:104-151, :90-100).
//
// Square numbering is C's: 0 = a1 ... 63 = h8, rank = sq/8, file = sq%8
// (bitboard.h:4).
package chess

import "math/bits"

// Colors and piece types, matching bitboard.h:16 and position.h:10.
const (
	White = 0
	Black = 1
)

const (
	Pawn = iota
	Knight
	Bishop
	Rook
	Queen
	King
	NoPiece
)

// Attack tables, built by initTables. Package-level and immutable after init,
// exactly like the C globals.
var (
	rookRelevantMask   [64]uint64
	bishopRelevantMask [64]uint64
	rookAttackTable    [rookAttackTableSize]uint64
	bishopAttackTable  [bishopAttackTableSize]uint64
	knightAttacksTbl   [64]uint64
	kingAttacksTbl     [64]uint64

	// Evaluation masks (bitboard.c:154-174).
	fileBB          [8]uint64
	adjacentFilesBB [8]uint64
	passedPawnMask  [2][64]uint64
)

var rookDirs = [4][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
var bishopDirs = [4][2]int{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}}

func popcount(x uint64) int { return bits.OnesCount64(x) }
func ctz(x uint64) int      { return bits.TrailingZeros64(x) }

// popLSB pops the lowest set bit and returns its index. x must be nonzero.
func popLSB(x *uint64) int {
	sq := ctz(*x)
	*x &= *x - 1
	return sq
}

// slidingAttacksSlow is bitboard.c:25-39: every ray square up to and INCLUDING
// the first occupied one. This is what fills the magic tables, so the magic
// lookups and this function agree by construction.
func slidingAttacksSlow(sq int, occ uint64, dirs [4][2]int) uint64 {
	var att uint64
	r0, f0 := sq/8, sq%8
	for d := 0; d < 4; d++ {
		r, f := r0+dirs[d][0], f0+dirs[d][1]
		for r >= 0 && r < 8 && f >= 0 && f < 8 {
			b := uint64(1) << uint(r*8+f)
			att |= b
			if occ&b != 0 {
				break
			}
			r += dirs[d][0]
			f += dirs[d][1]
		}
	}
	return att
}

// relevantMaskSlow is bitboard.c:41-55: the ray squares minus the outermost
// one in each direction (a blocker on the edge changes nothing).
func relevantMaskSlow(sq int, dirs [4][2]int) uint64 {
	var mask uint64
	r0, f0 := sq/8, sq%8
	for d := 0; d < 4; d++ {
		r, f := r0+dirs[d][0], f0+dirs[d][1]
		for {
			nr, nf := r+dirs[d][0], f+dirs[d][1]
			if nr < 0 || nr > 7 || nf < 0 || nf > 7 {
				break
			}
			mask |= uint64(1) << uint(r*8+f)
			r, f = nr, nf
		}
	}
	return mask
}

// buildMagicTables is bitboard.c:57-70, including the Carry-Rippler subset
// enumeration. Go's uint64 multiply wraps exactly as C's does, which is what
// the magic hash relies on.
func buildMagicTables(dirs [4][2]int, relMask *[64]uint64, magic *[64]uint64,
	shift *[64]int, offset *[64]int, table []uint64) {
	for sq := 0; sq < 64; sq++ {
		mask := relevantMaskSlow(sq, dirs)
		relMask[sq] = mask
		var sub uint64
		for {
			idx := (sub * magic[sq]) >> uint(shift[sq])
			table[offset[sq]+int(idx)] = slidingAttacksSlow(sq, sub, dirs)
			sub = (sub - mask) & mask
			if sub == 0 {
				break
			}
		}
	}
}

// buildLeaperTables is the knight/king half of bitboard.c:72-102. The pawn
// attack tables are omitted: eval_classical never consults them.
func buildLeaperTables() {
	knightD := [8][2]int{{2, 1}, {2, -1}, {-2, 1}, {-2, -1}, {1, 2}, {1, -2}, {-1, 2}, {-1, -2}}
	kingD := [8][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}}
	for sq := 0; sq < 64; sq++ {
		r, f := sq/8, sq%8
		var kn, kg uint64
		for i := 0; i < 8; i++ {
			if nr, nf := r+knightD[i][0], f+knightD[i][1]; nr >= 0 && nr < 8 && nf >= 0 && nf < 8 {
				kn |= uint64(1) << uint(nr*8+nf)
			}
			if nr, nf := r+kingD[i][0], f+kingD[i][1]; nr >= 0 && nr < 8 && nf >= 0 && nf < 8 {
				kg |= uint64(1) << uint(nr*8+nf)
			}
		}
		knightAttacksTbl[sq] = kn
		kingAttacksTbl[sq] = kg
	}
}

// buildEvalMasks is bitboard.c:154-174.
func buildEvalMasks() {
	for f := 0; f < 8; f++ {
		var m uint64
		for r := 0; r < 8; r++ {
			m |= uint64(1) << uint(r*8+f)
		}
		fileBB[f] = m
	}
	for f := 0; f < 8; f++ {
		adjacentFilesBB[f] = 0
		if f > 0 {
			adjacentFilesBB[f] |= fileBB[f-1]
		}
		if f < 7 {
			adjacentFilesBB[f] |= fileBB[f+1]
		}
	}
	for sq := 0; sq < 64; sq++ {
		r, f := sq/8, sq%8
		files := fileBB[f] | adjacentFilesBB[f]
		var aheadW, aheadB uint64
		for rr := r + 1; rr < 8; rr++ {
			aheadW |= uint64(0xFF) << uint(rr*8)
		}
		for rr := r - 1; rr >= 0; rr-- {
			aheadB |= uint64(0xFF) << uint(rr*8)
		}
		passedPawnMask[White][sq] = files & aheadW
		passedPawnMask[Black][sq] = files & aheadB
	}
}

// Sliding attack sets (bitboard.h:83-95). One multiply hashes the blockers on
// the relevant mask into this square's table slice.
func rookAttacks(sq int, occ uint64) uint64 {
	return rookAttackTable[rookAttackOffset[sq]+
		int(((occ&rookRelevantMask[sq])*rookMagic[sq])>>uint(rookShift[sq]))]
}

func bishopAttacks(sq int, occ uint64) uint64 {
	return bishopAttackTable[bishopAttackOffset[sq]+
		int(((occ&bishopRelevantMask[sq])*bishopMagic[sq])>>uint(bishopShift[sq]))]
}

func queenAttacks(sq int, occ uint64) uint64 {
	return rookAttacks(sq, occ) | bishopAttacks(sq, occ)
}

// init builds every table once, mirroring bitboard_init (bitboard.c:176-188).
// Go guarantees this runs before any exported function, so unlike the C there
// is no idempotence flag to forget to call.
func init() {
	buildMagicTables(rookDirs, &rookRelevantMask, &rookMagic, &rookShift,
		&rookAttackOffset, rookAttackTable[:])
	buildMagicTables(bishopDirs, &bishopRelevantMask, &bishopMagic, &bishopShift,
		&bishopAttackOffset, bishopAttackTable[:])
	buildLeaperTables()
	buildEvalMasks()
}
