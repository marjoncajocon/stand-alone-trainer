// Position and FEN, ported from chess-cli/c_port/position.{h,c}.
//
// Only the state eval_classical reads is kept. Zobrist keys are NOT computed:
// the C side needs them for its transposition table and for its trainer's
// dedup hash, but this trainer dedups on an EXACT packed key instead (see
// internal/corpus), so a 64-bit hash would only add collisions.

package chess

import "strconv"

// Castling-rights bits (position.h:13).
const (
	crWK = 1
	crWQ = 2
	crBK = 4
	crBQ = 8
)

// castleNone marks "this castling is unavailable" in CastleRookFile
// (position.h:46).
const castleNone = 0xFF

// Position mirrors the C struct (position.h:15-44) minus the search-only
// fields (key, nnue_acc).
type Position struct {
	Pieces   [2][6]uint64 // [WHITE/BLACK][PAWN..KING]
	Occ      [2]uint64
	OccAll   uint64
	Stm      int
	Castling uint8
	EpSq     int8 // -1 if none
	Halfmove uint8
	Fullmove uint16

	// Chess960 castling: the FILE (0-7) of each side's castling rook, or
	// castleNone. [color][0]=queenside/a-side, [1]=kingside/h-side. Constant
	// for a game; set at FEN parse (position.h:25-31). Two 960 positions with
	// identical placement but different castling rooks are genuinely different
	// positions, which is why the corpus dedup key carries these bytes.
	CastleRookFile [2][2]uint8
}

func (p *Position) refreshOcc() {
	for s := 0; s < 2; s++ {
		p.Occ[s] = 0
		for pc := 0; pc < 6; pc++ {
			p.Occ[s] |= p.Pieces[s][pc]
		}
	}
	p.OccAll = p.Occ[White] | p.Occ[Black]
}

// findCastleRook is position.c:78-88: for X-FEN 'K'/'Q', the outermost rook on
// the given side of the king. Returns the rook's file, or -1.
func (p *Position) findCastleRook(color, rank, kfile, dir int) int {
	rooks := p.Pieces[color][Rook] & (uint64(0xFF) << uint(rank*8))
	best := -1
	for rooks != 0 {
		f := popLSB(&rooks) % 8
		if dir > 0 && f > kfile && (best < 0 || f > best) {
			best = f
		}
		if dir < 0 && f < kfile && (best < 0 || f < best) {
			best = f
		}
	}
	return best
}

// atoiPrefix reads the leading decimal digits of s, as C's atoi does. Unlike
// strconv.Atoi it stops at the first non-digit instead of failing, which is
// what position.c:178-183 relies on when the clocks are followed by more text.
func atoiPrefix(s []byte) int {
	i := 0
	for i < len(s) && s[i] == ' ' {
		i++
	}
	neg := false
	if i < len(s) && (s[i] == '-' || s[i] == '+') {
		neg = s[i] == '-'
		i++
	}
	n := 0
	for i < len(s) && s[i] >= '0' && s[i] <= '9' {
		n = n*10 + int(s[i]-'0')
		if n > 1<<30 { // saturate rather than wrap on absurd input
			n = 1 << 30
			for i < len(s) && s[i] >= '0' && s[i] <= '9' {
				i++
			}
			break
		}
		i++
	}
	if neg {
		return -n
	}
	return n
}

// FromFEN parses a FEN into p, returning false on any malformed input.
//
// Prefer FromFENBytes on the hot path: this wrapper copies the string.
func (p *Position) FromFEN(fen string) bool {
	return p.FromFENBytes([]byte(fen))
}

// FromFENBytes is a direct port of pos_from_fen (position.c:90-197), including
// its validation: exactly one king per side, and no square occupied by both
// colors.
//
// It takes []byte rather than string because the corpus loader slices FENs
// straight out of the line buffer -- at ~50M lines a per-line string copy is
// pure garbage. The slice is never retained.
//
// Castling accepts BOTH standard KQkq and X-FEN / Shredder file letters
// (position.h:86-88). That is not optional here: Chess960 self-play data is
// pooled into the same corpus, and Shredder-form rights appear in it.
func (p *Position) FromFENBytes(fen []byte) bool {
	*p = Position{}
	p.EpSq = -1
	p.Fullmove = 1
	p.CastleRookFile[White][0] = castleNone
	p.CastleRookFile[White][1] = castleNone
	p.CastleRookFile[Black][0] = castleNone
	p.CastleRookFile[Black][1] = castleNone

	i := 0
	// 1. piece placement, rank 8 first
	r, f := 7, 0
	for ; i < len(fen) && fen[i] != ' '; i++ {
		c := fen[i]
		switch {
		case c == '/':
			if f != 8 || r == 0 {
				return false
			}
			r--
			f = 0
		case c >= '1' && c <= '8':
			f += int(c - '0')
			if f > 8 {
				return false
			}
		default:
			if f > 7 {
				return false
			}
			side := White
			if c >= 'a' {
				side = Black
			}
			var piece int
			switch c | 32 {
			case 'p':
				piece = Pawn
			case 'n':
				piece = Knight
			case 'b':
				piece = Bishop
			case 'r':
				piece = Rook
			case 'q':
				piece = Queen
			case 'k':
				piece = King
			default:
				return false
			}
			p.Pieces[side][piece] |= uint64(1) << uint(r*8+f)
			f++
		}
	}
	if r != 0 || f != 8 {
		return false
	}
	if i >= len(fen) || fen[i] != ' ' {
		return false
	}
	i++

	// 2. side to move
	if i >= len(fen) {
		return false
	}
	switch fen[i] {
	case 'w':
		p.Stm = White
	case 'b':
		p.Stm = Black
	default:
		return false
	}
	i++
	if i >= len(fen) || fen[i] != ' ' {
		return false
	}
	i++

	// 3. castling rights. Kings are already placed by section 1.
	if i < len(fen) && fen[i] == '-' {
		i++
	} else {
		kfile := [2]int{-1, -1}
		if p.Pieces[White][King] != 0 {
			kfile[White] = ctz(p.Pieces[White][King]) % 8
		}
		if p.Pieces[Black][King] != 0 {
			kfile[Black] = ctz(p.Pieces[Black][King]) % 8
		}
		for ; i < len(fen) && fen[i] != ' '; i++ {
			c := fen[i]
			color := White
			if c >= 'a' {
				color = Black
			}
			rank := 0
			if color == Black {
				rank = 7
			}
			kf := kfile[color]
			lc := c | 32
			var rookFile, side int
			switch {
			case lc == 'k':
				rookFile, side = p.findCastleRook(color, rank, kf, +1), 1
			case lc == 'q':
				rookFile, side = p.findCastleRook(color, rank, kf, -1), 0
			case lc >= 'a' && lc <= 'h':
				rookFile = int(lc - 'a')
				if rookFile > kf {
					side = 1
				} else {
					side = 0
				}
			default:
				return false
			}
			if rookFile < 0 || kf < 0 {
				return false
			}
			p.CastleRookFile[color][side] = uint8(rookFile)
			if color == White {
				if side != 0 {
					p.Castling |= crWK
				} else {
					p.Castling |= crWQ
				}
			} else {
				if side != 0 {
					p.Castling |= crBK
				} else {
					p.Castling |= crBQ
				}
			}
		}
	}
	if i >= len(fen) || fen[i] != ' ' {
		return false
	}
	i++

	// 4. en passant
	if i < len(fen) && fen[i] == '-' {
		i++
	} else {
		if i+1 >= len(fen) ||
			fen[i] < 'a' || fen[i] > 'h' || fen[i+1] < '1' || fen[i+1] > '8' {
			return false
		}
		p.EpSq = int8(int(fen[i+1]-'1')*8 + int(fen[i]-'a'))
		i += 2
	}

	// 5-6. clocks (optional; default 0 / 1)
	if i < len(fen) && fen[i] == ' ' {
		i++
		p.Halfmove = uint8(atoiPrefix(fen[i:]))
		for i < len(fen) && fen[i] != ' ' {
			i++
		}
		if i < len(fen) && fen[i] == ' ' {
			i++
			if fm := atoiPrefix(fen[i:]); fm > 0 {
				p.Fullmove = uint16(fm)
			}
		}
	}

	// Exactly one king each, and the two sides disjoint.
	if popcount(p.Pieces[White][King]) != 1 {
		return false
	}
	if popcount(p.Pieces[Black][King]) != 1 {
		return false
	}
	p.refreshOcc()
	if p.Occ[White]&p.Occ[Black] != 0 {
		return false
	}
	return true
}

var pieceChar = [2][6]byte{
	{'P', 'N', 'B', 'R', 'Q', 'K'},
	{'p', 'n', 'b', 'r', 'q', 'k'},
}

// ToFEN is pos_to_fen (position.c:199-260). Kept so the FEN round-trip can be
// gated against C -- a parser that silently drops a field would otherwise look
// fine right up until the anchor disagreed.
func (p *Position) ToFEN() string {
	out := make([]byte, 0, 100)
	for r := 7; r >= 0; r-- {
		empty := 0
		for f := 0; f < 8; f++ {
			sq := uint(r*8 + f)
			var c byte
			for s := 0; s < 2 && c == 0; s++ {
				for pc := 0; pc < 6; pc++ {
					if p.Pieces[s][pc]&(uint64(1)<<sq) != 0 {
						c = pieceChar[s][pc]
						break
					}
				}
			}
			if c == 0 {
				empty++
			} else {
				if empty != 0 {
					out = append(out, byte('0'+empty))
				}
				empty = 0
				out = append(out, c)
			}
		}
		if empty != 0 {
			out = append(out, byte('0'+empty))
		}
		if r != 0 {
			out = append(out, '/')
		}
	}
	out = append(out, ' ')
	if p.Stm == White {
		out = append(out, 'w')
	} else {
		out = append(out, 'b')
	}
	out = append(out, ' ')
	if p.Castling == 0 {
		out = append(out, '-')
	} else {
		// Standard-shaped rights (king on e, rook on a/h) print as KQkq, so a
		// standard FEN round-trips byte-identically; anything else is Shredder.
		wk, bk := -1, -1
		if p.Pieces[White][King] != 0 {
			wk = ctz(p.Pieces[White][King]) % 8
		}
		if p.Pieces[Black][King] != 0 {
			bk = ctz(p.Pieces[Black][King]) % 8
		}
		std := true
		if p.Castling&crWK != 0 && !(wk == 4 && p.CastleRookFile[White][1] == 7) {
			std = false
		}
		if p.Castling&crWQ != 0 && !(wk == 4 && p.CastleRookFile[White][0] == 0) {
			std = false
		}
		if p.Castling&crBK != 0 && !(bk == 4 && p.CastleRookFile[Black][1] == 7) {
			std = false
		}
		if p.Castling&crBQ != 0 && !(bk == 4 && p.CastleRookFile[Black][0] == 0) {
			std = false
		}
		if std {
			if p.Castling&crWK != 0 {
				out = append(out, 'K')
			}
			if p.Castling&crWQ != 0 {
				out = append(out, 'Q')
			}
			if p.Castling&crBK != 0 {
				out = append(out, 'k')
			}
			if p.Castling&crBQ != 0 {
				out = append(out, 'q')
			}
		} else {
			if p.Castling&crWK != 0 {
				out = append(out, 'A'+p.CastleRookFile[White][1])
			}
			if p.Castling&crWQ != 0 {
				out = append(out, 'A'+p.CastleRookFile[White][0])
			}
			if p.Castling&crBK != 0 {
				out = append(out, 'a'+p.CastleRookFile[Black][1])
			}
			if p.Castling&crBQ != 0 {
				out = append(out, 'a'+p.CastleRookFile[Black][0])
			}
		}
	}
	out = append(out, ' ')
	if p.EpSq < 0 {
		out = append(out, '-')
	} else {
		out = append(out, byte('a'+p.EpSq%8), byte('1'+p.EpSq/8))
	}
	out = append(out, ' ')
	out = strconv.AppendInt(out, int64(p.Halfmove), 10)
	out = append(out, ' ')
	out = strconv.AppendInt(out, int64(p.Fullmove), 10)
	return string(out)
}
