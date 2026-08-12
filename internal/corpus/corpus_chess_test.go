package corpus_test

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"nnuetrainer/internal/chess"
	"nnuetrainer/internal/corpus"
	"nnuetrainer/internal/variant"
)

func writeCorpus(t *testing.T, lines ...string) string {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, "selfplay_chess_test_w0.txt")
	var buf []byte
	for _, l := range lines {
		buf = append(buf, l...)
		buf = append(buf, '\n')
	}
	if err := os.WriteFile(p, buf, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func loadChess(t *testing.T, lines ...string) *corpus.Set {
	t.Helper()
	v, err := variant.Get("chess")
	if err != nil {
		t.Fatal(err)
	}
	set, err := corpus.Load(v, corpus.Options{
		Globs:   []string{writeCorpus(t, lines...)},
		Holdout: 0, // everything into train, so the test sees every sample
		Lambda:  0, // outcome only unless a test overrides
		KScale:  1.0 / 400,
		Quiet:   true,
	})
	if err != nil {
		t.Fatal(err)
	}
	return set
}

// TestChessPerspectiveFlip is the single most important test in this package.
//
// The corpus records WHITE's outcome and a WHITE-perspective eval, but the net
// output, the eval_classical anchor and therefore the training target are all
// STM-relative (trainer.c:203, :206). If the flip is missing or inverted the
// trainer still runs, the MSE still falls, the parity gates still pass -- and
// the net is trained on labels that are backwards for every black-to-move
// position, which is half the corpus. There is no other symptom.
func TestChessPerspectiveFlip(t *testing.T) {
	const wFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
	const bFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"

	// The same game result, "1-0" (White won), seen from each side. Loaded one
	// line at a time so each target is unambiguously attributable.
	wY := onlySample(t, loadChess(t, wFEN+" 1-0 0")).Y
	bY := onlySample(t, loadChess(t, bFEN+" 1-0 0")).Y

	if wY != 1.0 {
		t.Errorf("white to move, White won: target %v, want 1.0", wY)
	}
	if bY != 0.0 {
		t.Errorf("black to move, White won: target %v, want 0.0 "+
			"(the target is STM-relative -- Black lost)", bY)
	}

	// And a draw must stay 0.5 from both sides.
	for _, fen := range []string{wFEN, bFEN} {
		s := onlySample(t, loadChess(t, fen+" 1/2-1/2 0"))
		if s.Y != 0.5 {
			t.Errorf("draw target %v, want 0.5 (%s)", s.Y, fen)
		}
	}
}

// TestChessEvalFlip checks the eval column is flipped with the same rule. The
// eval is blended into the target by lambda, so an unflipped eval poisons the
// label even when the outcome half is right.
func TestChessEvalFlip(t *testing.T) {
	const bFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
	v, _ := variant.Get("chess")

	// lambda 1.0 => the target is entirely sigmoid(eval/400).
	set, err := corpus.Load(v, corpus.Options{
		Globs:   []string{writeCorpus(t, bFEN+" 1/2-1/2 0 400")},
		Holdout: 0,
		Lambda:  1.0,
		KScale:  1.0 / 400,
		Quiet:   true,
	})
	if err != nil {
		t.Fatal(err)
	}
	got := onlySample(t, set).Y

	// +400 cp is WHITE's view; Black to move sees -400.
	want := 1 / (1 + math.Exp(400.0/400.0))
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("eval target %v, want %v (a +400 white eval is -400 to the "+
			"black mover; an unflipped eval would give %v)",
			got, want, 1/(1+math.Exp(-1.0)))
	}
}

// TestChessDedupKeepsStm guards the difference from the draughts loader, whose
// dedup key is the board ALONE because that kernel has no stm input.
func TestChessDedupKeepsStm(t *testing.T) {
	const wFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
	const bFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
	set := loadChess(t, wFEN+" 1-0 0", bFEN+" 1-0 0", wFEN+" 1-0 1")
	if set.Stats.Unique != 2 {
		t.Errorf("unique = %d, want 2 (stm distinguishes, the repeat merges)",
			set.Stats.Unique)
	}
	if set.Stats.Lines != 3 {
		t.Errorf("lines = %d, want 3", set.Stats.Lines)
	}
}

// TestChessAnchorIsClassicalEval checks the loader stores eval_classical and
// not something else -- a material count, or zero.
func TestChessAnchorIsClassicalEval(t *testing.T) {
	const fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
	var pos chess.Position
	if !pos.FromFEN(fen) {
		t.Fatal("fen failed to parse")
	}
	want := pos.AnchorCP()

	s := onlySample(t, loadChess(t, fen+" 1/2-1/2 0"))
	if int(s.Anchor) != want {
		t.Errorf("anchor = %d, want eval_classical = %d", s.Anchor, want)
	}
	if want == 0 {
		t.Error("this test position has a zero anchor, so it proves nothing; pick another")
	}
}

// TestChessMalformedLinesSkipped: a bad FEN or an unknown result token must be
// dropped, not counted or defaulted.
func TestChessMalformedLinesSkipped(t *testing.T) {
	const good = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 1-0 0"
	set := loadChess(t,
		good,
		"not/a/fen 8/8/8/8 w KQkq - 0 1 1-0 0", // unparseable placement
		"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 * 0", // '*' result
		"too few fields",
	)
	if set.Stats.Lines != 1 {
		t.Errorf("lines = %d, want 1 (three lines are malformed)", set.Stats.Lines)
	}
	if set.Stats.Unique != 1 {
		t.Errorf("unique = %d, want 1", set.Stats.Unique)
	}
}

// TestChessFeaturesInRange guards the invariant the model relies on.
func TestChessFeaturesInRange(t *testing.T) {
	set := loadChess(t,
		"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 1-0 0",
		"8/8/8/4k3/8/8/4K3/8 b - - 0 1 0-1 1",
	)
	for _, f := range set.Pool {
		if int(f) >= chess.NumFeatures {
			t.Fatalf("feature %d >= NumFeatures %d", f, chess.NumFeatures)
		}
	}
	for _, s := range set.Train {
		if n := s.End - s.Off; n > chess.MaxFeatures {
			t.Fatalf("sample has %d features, more than MaxFeatures %d", n, chess.MaxFeatures)
		}
	}
}

func onlySample(t *testing.T, s *corpus.Set) corpus.Sample {
	t.Helper()
	all := append(append([]corpus.Sample{}, s.Train...), s.Valid...)
	if len(all) != 1 {
		t.Fatalf("got %d samples, want exactly 1", len(all))
	}
	return all[0]
}
