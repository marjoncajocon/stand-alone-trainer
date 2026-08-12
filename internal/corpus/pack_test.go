package corpus_test

import (
	"os"
	"path/filepath"
	"testing"

	"nnuetrainer/internal/corpus"
	"nnuetrainer/internal/variant"
)

// TestPackRoundTripChess: text -> .ntc -> read must reproduce the set exactly.
// The packed corpus is what actually travels to Colab, so anything the pack
// drops is invisible locally and wrong remotely.
func TestPackRoundTripChess(t *testing.T) {
	v, err := variant.Get("chess")
	if err != nil {
		t.Fatal(err)
	}
	lines := []string{
		"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 1-0 0 34",
		"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1 1-0 0 -12",
		"r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3 0-1 1",
		"8/8/8/4k3/8/8/4K3/8 w - - 0 1 1/2-1/2 2",
	}
	want, err := corpus.Load(v, corpus.Options{
		Globs:   []string{writeCorpus(t, lines...)},
		Holdout: 2,
		Lambda:  0.5,
		KScale:  1.0 / 400,
		Quiet:   true,
	})
	if err != nil {
		t.Fatal(err)
	}

	p := filepath.Join(t.TempDir(), "chess.ntc")
	if err := want.Save(p); err != nil {
		t.Fatal(err)
	}
	got, err := corpus.LoadPacked(v, p, true)
	if err != nil {
		t.Fatal(err)
	}

	if got.Stats.Lines != want.Stats.Lines || got.Stats.Unique != want.Stats.Unique {
		t.Errorf("counters drift: got %d/%d want %d/%d",
			got.Stats.Lines, got.Stats.Unique, want.Stats.Lines, want.Stats.Unique)
	}
	if got.Stats.KScale != want.Stats.KScale {
		t.Errorf("KScale drift: got %v want %v (chess must carry 1/400, not the "+
			"draughts 1/256, or Colab silently trains to the wrong sigmoid)",
			got.Stats.KScale, want.Stats.KScale)
	}
	if len(got.Pool) != len(want.Pool) {
		t.Fatalf("pool length %d != %d", len(got.Pool), len(want.Pool))
	}
	for i := range want.Pool {
		if got.Pool[i] != want.Pool[i] {
			t.Fatalf("pool[%d] drift", i)
		}
	}
	if len(got.Train) != len(want.Train) || len(got.Valid) != len(want.Valid) {
		t.Fatalf("split drift: got %d/%d want %d/%d",
			len(got.Train), len(got.Valid), len(want.Train), len(want.Valid))
	}
	for i := range want.Train {
		if got.Train[i] != want.Train[i] {
			t.Fatalf("train[%d] drift: %+v vs %+v", i, got.Train[i], want.Train[i])
		}
	}
	for i := range want.Valid {
		if got.Valid[i] != want.Valid[i] {
			t.Fatalf("valid[%d] drift: %+v vs %+v", i, got.Valid[i], want.Valid[i])
		}
	}
}

// TestPackRejectsKernelMismatch: a corpus packed for one kernel must not load
// under the other. Geometry alone would not catch it -- turkish is 256 features
// and chess is 768, but a future draughts variant could collide.
func TestPackRejectsKernelMismatch(t *testing.T) {
	chessV, _ := variant.Get("chess")
	filV, _ := variant.Get("filipino")

	set, err := corpus.Load(chessV, corpus.Options{
		Globs:   []string{writeCorpus(t, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 1-0 0")},
		Holdout: 0, Lambda: 0.5, KScale: 1.0 / 400, Quiet: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(t.TempDir(), "chess.ntc")
	if err := set.Save(p); err != nil {
		t.Fatal(err)
	}
	if _, err := corpus.LoadPacked(filV, p, true); err == nil {
		t.Error("loading a chess corpus as filipino succeeded; it must fail")
	}
}

// TestPackRejectsTruncated guards the trailing-byte check on the new layout.
func TestPackRejectsTruncated(t *testing.T) {
	v, _ := variant.Get("chess")
	set, err := corpus.Load(v, corpus.Options{
		Globs:   []string{writeCorpus(t, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 1-0 0")},
		Holdout: 0, Lambda: 0.5, KScale: 1.0 / 400, Quiet: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(t.TempDir(), "chess.ntc")
	if err := set.Save(p); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, raw[:len(raw)-3], 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := corpus.LoadPacked(v, p, true); err == nil {
		t.Error("a truncated corpus loaded without error")
	}
}
