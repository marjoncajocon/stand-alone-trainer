package corpus_test

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"nnuetrainer/internal/corpus"
	"nnuetrainer/internal/variant"
)

// Playable indices on an 8x8 draughts board are the dark squares,
// (row+col)%2 == 1 (variant.SquareOf). Putting a piece anywhere else would make
// AppendFeats skip it and the test would be measuring the wrong thing.
var darkSquares = []int{
	1, 3, 5, 7, 8, 10, 12, 14, 17, 19, 21, 23, 24, 26, 28, 30,
	33, 35, 37, 39, 40, 42, 44, 46, 49, 51, 53, 55, 56, 58, 60, 62,
}

// fboard renders a 64-cell filipino board with pieces at the given dark-square
// slots, taken in order from darkSquares.
func fboard(pieces string, slots ...int) string {
	b := make([]byte, 64)
	for i := range b {
		b[i] = '.'
	}
	for i, s := range slots {
		b[darkSquares[s]] = pieces[i]
	}
	return string(b)
}

// writeAt writes one corpus file into dir. corpus_chess_test.go's writeCorpus
// makes its own TempDir with one fixed name, which cannot express the multi-file
// cases the counter exists for.
func writeAt(t *testing.T, dir, name string, lines ...string) string {
	t.Helper()
	p := filepath.Join(dir, name)
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

func getVariant(t *testing.T, name string) *variant.Variant {
	t.Helper()
	v, err := variant.Get(name)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func countOK(t *testing.T, v *variant.Variant, o corpus.CountOptions) *corpus.CountResult {
	t.Helper()
	r, err := corpus.Count(v, o)
	if err != nil {
		t.Fatal(err)
	}
	if r.Errs != 0 {
		t.Fatalf("Count reported %d file errors", r.Errs)
	}
	return r
}

// Boards used across the tests. TB for filipino is total pieces <= 5.
var (
	bigA = fboard("XXXOOO", 0, 1, 2, 3, 4, 5)   // 6 pieces: survives
	bigB = fboard("XXXOOO", 6, 7, 8, 9, 10, 11) // 6 pieces, different squares
	tbA  = fboard("XXO", 0, 1, 2)               // 3 pieces: TB territory
)

// TestCountMatchesLoad is the test this whole package hangs on: an inventory
// that disagreed with what training actually reads would be worse than none.
//
// The corpus deliberately contains duplicate boards, TB-territory boards,
// malformed lines and repeated gameids, because those are exactly the four
// places the two code paths could drift.
func TestCountMatchesLoad(t *testing.T) {
	cases := []struct {
		variant string
		files   map[string][]string
		kScale  float64
	}{{
		variant: "filipino",
		files: map[string][]string{
			"selfplay_filipino_a_w0.txt": {
				bigA + " A X 0 12",
				bigA + " P O 0 -12", // duplicate board, same game
				bigB + " A D 1",
				tbA + " A X 1", // in LINES, excluded from UNIQUE
				"short A X 0",  // malformed: wrong cell count
				"two fields",   // malformed: too few fields
			},
			"selfplay_filipino_a_w1.txt": {
				bigA + " A X 0", // same board as w0: one unique overall
				bigB + " P O 0",
			},
		},
		kScale: 1.0 / 256,
	}, {
		variant: "chess",
		files: map[string][]string{
			"selfplay_chess_a_w0.txt": {
				"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 1-0 0 31",
				"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1 1-0 0",
				"8/8/8/4k3/8/8/4K3/8 w - - 0 1 1/2-1/2 1",
				"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 X-Y 2", // bad result
				"8/8/8/4k3/8/8/4K3/8 w - - 0 1 1-0",                              // no gameid: 7 fields
				"not even a fen",
			},
			"selfplay_chess_a_w1.txt": {
				"8/8/8/4k3/8/8/4K3/8 w - - 0 1 0-1 0", // duplicate of w0 line 3
			},
		},
		kScale: 1.0 / 400,
	}}

	for _, tc := range cases {
		for _, threads := range []int{1, 4} {
			t.Run(tc.variant, func(t *testing.T) {
				v := getVariant(t, tc.variant)
				dir := t.TempDir()
				for name, lines := range tc.files {
					writeAt(t, dir, name, lines...)
				}
				glob := filepath.Join(dir, "*.txt")

				set, err := corpus.Load(v, corpus.Options{
					Globs: []string{glob}, Holdout: 0, Lambda: 0,
					KScale: tc.kScale, Quiet: true,
				})
				if err != nil {
					t.Fatal(err)
				}
				got := countOK(t, v, corpus.CountOptions{
					Globs: []string{glob}, Threads: threads,
					Unique: true, Exact: true,
				})

				if got.Lines != set.Stats.Lines {
					t.Errorf("threads=%d Lines = %d, Load says %d", threads, got.Lines, set.Stats.Lines)
				}
				if got.Unique != set.Stats.Unique {
					t.Errorf("threads=%d Unique = %d, Load says %d", threads, got.Unique, set.Stats.Unique)
				}
				if got.TBSkip != set.Stats.TBSkip {
					t.Errorf("threads=%d TBSkip = %d, Load says %d", threads, got.TBSkip, set.Stats.TBSkip)
				}
				if got.Files != len(tc.files) {
					t.Errorf("Files = %d, want %d", got.Files, len(tc.files))
				}
			})
		}
	}
}

// TestCountMalformedVsTBSkip pins the distinction the whole report hangs on:
// MALFORMED lines are rejected before Load counts them and are NOT in LINES,
// while TB-SKIP lines ARE in LINES and are only excluded from UNIQUE.
func TestCountMalformedVsTBSkip(t *testing.T) {
	v := getVariant(t, "filipino")
	dir := t.TempDir()
	writeAt(t, dir, "selfplay_filipino_x_w0.txt",
		bigA+" A X 0",      // clean
		bigA[:63]+" A X 0", // 63 cells: malformed
		"only two",         // malformed
		tbA+" A X 1",       // accepted, TB territory
	)
	got := countOK(t, v, corpus.CountOptions{
		Globs: []string{filepath.Join(dir, "*.txt")}, Unique: true,
	})
	if got.Lines != 2 {
		t.Errorf("Lines = %d, want 2 (clean + TB)", got.Lines)
	}
	if got.Malformed != 2 {
		t.Errorf("Malformed = %d, want 2", got.Malformed)
	}
	if got.TBSkip != 1 {
		t.Errorf("TBSkip = %d, want 1", got.TBSkip)
	}
	if got.Unique != 1 {
		t.Errorf("Unique = %d, want 1 (the TB board is excluded)", got.Unique)
	}
}

// TestCountGamesPerFile locks in per-file gameid scoping: gameids restart at 0
// in every worker file, so two files each holding games 0,1,2 hold six games.
func TestCountGamesPerFile(t *testing.T) {
	v := getVariant(t, "filipino")
	dir := t.TempDir()
	for _, name := range []string{"selfplay_filipino_x_w0.txt", "selfplay_filipino_x_w1.txt"} {
		writeAt(t, dir, name,
			bigA+" A X 0", bigA+" P O 0",
			bigB+" A D 1",
			bigA+" A X 2",
		)
	}
	got := countOK(t, v, corpus.CountOptions{
		Globs: []string{filepath.Join(dir, "*.txt")}, Unique: true, PerFile: true,
	})
	if got.Games != 6 {
		t.Errorf("Games = %d, want 6 (3 per file x 2 files, NOT 3)", got.Games)
	}
	for _, fc := range got.PerFile {
		if fc.Games != 3 {
			t.Errorf("%s: Games = %d, want 3", filepath.Base(fc.Path), fc.Games)
		}
	}
}

// TestCountGamesIncludeTBSkipped pins the one deliberate divergence from Load:
// a game whose every line is TB territory still counts as a game, because GAMES
// answers "what is in this corpus", not "what will training use". It is also
// what keeps GAMES comparable to the generator's .log "done:" count.
func TestCountGamesIncludeTBSkipped(t *testing.T) {
	v := getVariant(t, "filipino")
	dir := t.TempDir()
	writeAt(t, dir, "selfplay_filipino_tb_w0.txt", tbA+" A X 7")
	got := countOK(t, v, corpus.CountOptions{
		Globs: []string{filepath.Join(dir, "*.txt")}, Unique: true,
	})
	if got.Lines != 1 || got.TBSkip != 1 || got.Unique != 0 || got.Games != 1 {
		t.Errorf("Lines/TBSkip/Unique/Games = %d/%d/%d/%d, want 1/1/0/1",
			got.Lines, got.TBSkip, got.Unique, got.Games)
	}
}

// TestCountHashMatchesExact catches the realistic bug in the hashed unique set:
// hashing the wrong prefix length, or hashing before packKey filled the array.
func TestCountHashMatchesExact(t *testing.T) {
	for _, name := range []string{"filipino", "international", "chess"} {
		t.Run(name, func(t *testing.T) {
			v := getVariant(t, name)
			dir := t.TempDir()
			if v.IsChess() {
				writeAt(t, dir, "selfplay_chess_h_w0.txt",
					"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 1-0 0",
					"8/8/8/4k3/8/8/4K3/8 w - - 0 1 1/2-1/2 0",
					"8/8/8/4k3/8/8/4K3/8 b - - 0 1 1/2-1/2 1",
					"8/8/8/4k3/8/8/4K3/8 w - - 0 1 0-1 1",
				)
			} else {
				// Boards are rendered at this variant's own cell count, so the
				// 100-cell case really does exercise a 50-byte key.
				b := make([]byte, v.Cells)
				for i := range b {
					b[i] = '.'
				}
				var lines []string
				for i := 0; i < 8; i++ {
					c := append([]byte(nil), b...)
					// Six pieces, walked along the board so every line is a
					// distinct position and none is TB territory.
					for j := 0; j < 6; j++ {
						c[darkSquares[(i+j)%len(darkSquares)]] = "XXXOOO"[j]
					}
					lines = append(lines, string(c)+" A X 0")
				}
				lines = append(lines, lines[0]) // one exact duplicate
				writeAt(t, dir, "selfplay_"+name+"_h_w0.txt", lines...)
			}
			glob := filepath.Join(dir, "*.txt")
			exact := countOK(t, v, corpus.CountOptions{Globs: []string{glob}, Unique: true, Exact: true})
			hashed := countOK(t, v, corpus.CountOptions{Globs: []string{glob}, Unique: true})
			if exact.Unique != hashed.Unique {
				t.Errorf("Unique: exact %d != hashed %d", exact.Unique, hashed.Unique)
			}
			if exact.Unique == 0 {
				t.Fatal("test corpus produced no unique positions; it is not testing anything")
			}
		})
	}
}

// TestCountDeterministicUnderThreads is the scheduling-independence claim, tested.
func TestCountDeterministicUnderThreads(t *testing.T) {
	v := getVariant(t, "filipino")
	dir := t.TempDir()
	for i := 0; i < 6; i++ {
		writeAt(t, dir, "selfplay_filipino_d_w"+string(rune('0'+i))+".txt",
			bigA+" A X 0", bigB+" P O 0", tbA+" A D 1", bigA+" A X 1",
		)
	}
	opts := func(threads int) corpus.CountOptions {
		return corpus.CountOptions{
			Globs:   []string{filepath.Join(dir, "*.txt")},
			Threads: threads, Unique: true, PerFile: true,
		}
	}
	want := countOK(t, v, opts(1))
	for _, threads := range []int{2, 8} {
		got := countOK(t, v, opts(threads))
		if !reflect.DeepEqual(want, got) {
			t.Errorf("threads=%d result differs from threads=1:\n got %+v\nwant %+v", threads, got, want)
		}
	}
}

// TestCountNoGameID covers legacy 3-column draughts data, and pins the fact that
// the two formats disagree about a missing gameid: Load's chess branch needs 8
// fields, so a chess line without one is MALFORMED rather than gameid-less.
func TestCountNoGameID(t *testing.T) {
	v := getVariant(t, "filipino")
	dir := t.TempDir()
	writeAt(t, dir, "selfplay_filipino_g_w0.txt", bigA+" A X")
	got := countOK(t, v, corpus.CountOptions{Globs: []string{filepath.Join(dir, "*.txt")}, Unique: true})
	if got.Lines != 1 || got.NoGameID != 1 || got.Games != 0 {
		t.Errorf("draughts: Lines/NoGameID/Games = %d/%d/%d, want 1/1/0",
			got.Lines, got.NoGameID, got.Games)
	}

	cv := getVariant(t, "chess")
	cdir := t.TempDir()
	writeAt(t, cdir, "selfplay_chess_g_w0.txt", "8/8/8/4k3/8/8/4K3/8 w - - 0 1 1-0")
	cgot := countOK(t, cv, corpus.CountOptions{Globs: []string{filepath.Join(cdir, "*.txt")}, Unique: true})
	if cgot.Lines != 0 || cgot.Malformed != 1 || cgot.NoGameID != 0 {
		t.Errorf("chess: Lines/Malformed/NoGameID = %d/%d/%d, want 0/1/0",
			cgot.Lines, cgot.Malformed, cgot.NoGameID)
	}
}

// TestCountUniqueDisabled checks that --unique=false only drops UNIQUE.
func TestCountUniqueDisabled(t *testing.T) {
	v := getVariant(t, "filipino")
	dir := t.TempDir()
	writeAt(t, dir, "selfplay_filipino_u_w0.txt",
		bigA+" A X 0", bigA+" P O 0", bigB+" A D 1", tbA+" A X 1", "bad line",
	)
	glob := filepath.Join(dir, "*.txt")
	with := countOK(t, v, corpus.CountOptions{Globs: []string{glob}, Unique: true})
	without := countOK(t, v, corpus.CountOptions{Globs: []string{glob}})
	if without.Unique != -1 {
		t.Errorf("Unique = %d, want -1 when unique counting is off", without.Unique)
	}
	if without.Dedup() != 0 {
		t.Errorf("Dedup() = %v, want 0 when unique counting is off", without.Dedup())
	}
	without.Unique, without.Exact = with.Unique, with.Exact
	if !reflect.DeepEqual(with, without) {
		t.Errorf("disabling unique changed another counter:\n got %+v\nwant %+v", without, with)
	}
}

// TestPackedHeaderMatchesLoadPacked is the only thing keeping the shared .ntc
// header parser honest: cmd/count reads a packed corpus through PackedHeader
// while the trainer reads it through LoadPacked.
func TestPackedHeaderMatchesLoadPacked(t *testing.T) {
	v := getVariant(t, "chess")
	set := loadChess(t,
		"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 1-0 0 31",
		"8/8/8/4k3/8/8/4K3/8 b - - 0 1 1/2-1/2 1",
	)
	p := filepath.Join(t.TempDir(), "x.ntc")
	if err := set.Save(p); err != nil {
		t.Fatal(err)
	}

	info, err := corpus.PackedHeader(p)
	if err != nil {
		t.Fatal(err)
	}
	full, err := corpus.LoadPacked(v, p, true)
	if err != nil {
		t.Fatal(err)
	}
	if info.Name != v.Name || info.Cells != v.Cells || info.NumFeat != v.NumFeat {
		t.Errorf("identity: got %s/%d/%d, want %s/%d/%d",
			info.Name, info.Cells, info.NumFeat, v.Name, v.Cells, v.NumFeat)
	}
	if !info.HasKernel || info.Kernel != v.Kernel {
		t.Errorf("Kernel = %v (has=%v), want %v", info.Kernel, info.HasKernel, v.Kernel)
	}
	if info.Stats != full.Stats {
		t.Errorf("Stats from header = %+v, from LoadPacked = %+v", info.Stats, full.Stats)
	}
	if info.PoolLen != int64(len(full.Pool)) {
		t.Errorf("PoolLen = %d, want %d", info.PoolLen, len(full.Pool))
	}

	// A truncated header must be an error, not a panic: a packed corpus is the
	// artifact that travels to Colab and back, so a half-copied file is real.
	raw, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	short := filepath.Join(t.TempDir(), "short.ntc")
	if err := os.WriteFile(short, raw[:40], 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := corpus.PackedHeader(short); err == nil {
		t.Error("PackedHeader accepted a 40-byte file")
	}
}
