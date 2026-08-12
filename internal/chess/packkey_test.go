package chess

import (
	"bytes"
	"testing"
)

func keyOf(t *testing.T, fen string) []byte {
	t.Helper()
	var p Position
	if !p.FromFEN(fen) {
		t.Fatalf("failed to parse: %s", fen)
	}
	k := make([]byte, PackKeyLen)
	p.PackKey(k)
	return k
}

// TestPackKeyDistinguishes pins what the dedup key must and must not merge.
// Every "differs" case here is a position pair the net evaluates differently;
// merging one into the other would average two unrelated targets into a single
// training sample, which is silent and unrecoverable.
func TestPackKeyDistinguishes(t *testing.T) {
	const start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

	t.Run("clocks are ignored", func(t *testing.T) {
		// The net cannot see the clocks, so lines differing only there are the
		// same sample. This mirrors what the zobrist key does.
		a := keyOf(t, start)
		b := keyOf(t, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 7 42")
		if !bytes.Equal(a, b) {
			t.Error("halfmove/fullmove changed the key; they must not")
		}
	})

	t.Run("side to move differs", func(t *testing.T) {
		// Chess features are stm-RELATIVE, so this really is a different input.
		a := keyOf(t, start)
		b := keyOf(t, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
		if bytes.Equal(a, b) {
			t.Error("side to move did not change the key")
		}
	})

	t.Run("castling rights differ", func(t *testing.T) {
		a := keyOf(t, start)
		b := keyOf(t, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w Kkq - 0 1")
		if bytes.Equal(a, b) {
			t.Error("castling rights did not change the key")
		}
	})

	t.Run("en passant file differs", func(t *testing.T) {
		a := keyOf(t, "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2")
		b := keyOf(t, "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2")
		if bytes.Equal(a, b) {
			t.Error("en-passant target did not change the key")
		}
	})

	t.Run("placement differs", func(t *testing.T) {
		// Note the rights must drop 'K' too: with no rook on h1 the engine
		// rejects the FEN outright (position.c:157), which is correct.
		a := keyOf(t, start)
		b := keyOf(t, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w Qkq - 0 1")
		if bytes.Equal(a, b) {
			t.Error("a removed rook did not change the key")
		}
	})

	t.Run("960 castling rooks differ under identical placement", func(t *testing.T) {
		// The case that only matters because standard and 960 data are POOLED.
		// Same men on the same squares, same rights nibble, different castling
		// rooks -- genuinely different positions.
		a := keyOf(t, "rk2r3/pppppppp/8/8/8/8/PPPPPPPP/RK2R3 w Aa - 0 1")
		b := keyOf(t, "rk2r3/pppppppp/8/8/8/8/PPPPPPPP/RK2R3 w Ee - 0 1")
		if bytes.Equal(a, b) {
			t.Error("different castling rook files produced the same key")
		}
	})
}

// TestPackKeyEveryPiece checks the 4-bit nibble packing addresses all 64
// squares and both nibble halves. An off-by-one in the odd/even branch would
// otherwise merge positions that differ only on odd squares.
func TestPackKeyEveryPiece(t *testing.T) {
	base := keyOf(t, "8/8/8/8/8/8/8/K6k w - - 0 1")
	seen := map[string]bool{string(base): true}
	for sq := 0; sq < 64; sq++ {
		if sq == 0 || sq == 7 {
			continue // the two kings
		}
		var p Position
		if !p.FromFEN("8/8/8/8/8/8/8/K6k w - - 0 1") {
			t.Fatal("base position failed to parse")
		}
		p.Pieces[White][Queen] |= uint64(1) << uint(sq)
		p.refreshOcc()
		k := make([]byte, PackKeyLen)
		p.PackKey(k)
		if seen[string(k)] {
			t.Fatalf("a white queen on square %d collided with an earlier key", sq)
		}
		seen[string(k)] = true
	}
	if len(seen) != 63 {
		t.Fatalf("got %d distinct keys, want 63", len(seen))
	}
}
