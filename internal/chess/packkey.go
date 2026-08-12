package chess

// PackKeyLen is how many bytes PackKey writes. It fits inside the corpus
// loader's existing fixed-size key array, so chess needs no wider map key than
// the 10x10 draughts board already required.
//
//	0..31  the 64 squares, 4 bits each: 0 empty, else 1 + piece_type + 6*color
//	   32  side to move
//	   33  castling-rights nibble
//	   34  en-passant FILE + 1 (0 = none)
//	   35  white's castling rook files, one nibble each
//	   36  black's castling rook files, one nibble each
const PackKeyLen = 37

// PackKey writes an EXACT identity for this position into dst, which must have
// room for PackKeyLen bytes.
//
// chess-cli's own trainer dedups on the 64-bit zobrist key
// (tools/trainer.c:66-75). This is deliberately different: across the ~50M
// positions in the corpus a 64-bit hash has a real (if small) collision
// probability, and a collision silently MERGES two unrelated positions into one
// training sample with an averaged target. An exact key cannot do that, and it
// costs nothing here because the map key was already a fixed-size array.
//
// What is included is exactly what the zobrist includes, for the same reasons:
//
//   - The HALFMOVE and FULLMOVE clocks are excluded. They are not part of the
//     position for evaluation purposes and the net cannot see them, so two
//     lines differing only in the clocks are genuinely the same sample.
//   - Side to move IS included. Unlike the draughts kernel -- whose comment at
//     corpus.go notes it has no stm input -- chess features are stm-RELATIVE,
//     so the same placement with the other side to move is a different input.
//   - The castling ROOK FILES are included, not just the rights nibble. In
//     Chess960 two positions can share placement and rights yet have different
//     castling rooks; since standard and 960 data are pooled into one corpus,
//     leaving these out would merge positions that are not the same.
func (p *Position) PackKey(dst []byte) {
	_ = dst[PackKeyLen-1] // bounds check once

	for i := 0; i < 32; i++ {
		dst[i] = 0
	}
	for c := 0; c < 2; c++ {
		for pt := Pawn; pt <= King; pt++ {
			bb := p.Pieces[c][pt]
			for bb != 0 {
				sq := popLSB(&bb)
				code := byte(1 + pt + 6*c)
				if sq&1 == 0 {
					dst[sq>>1] |= code << 4
				} else {
					dst[sq>>1] |= code
				}
			}
		}
	}
	dst[32] = byte(p.Stm)
	dst[33] = p.Castling
	if p.EpSq < 0 {
		dst[34] = 0
	} else {
		dst[34] = byte(p.EpSq%8) + 1
	}
	// castleNone (0xFF) folds to nibble 0xF, which no real file (0-7) uses.
	dst[35] = ((p.CastleRookFile[White][0] & 0xF) << 4) | (p.CastleRookFile[White][1] & 0xF)
	dst[36] = ((p.CastleRookFile[Black][0] & 0xF) << 4) | (p.CastleRookFile[Black][1] & 0xF)
}
