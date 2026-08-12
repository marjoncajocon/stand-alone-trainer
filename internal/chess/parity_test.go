package chess

import (
	"bufio"
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

// TestAnchorParity is the gate that makes the eval_classical port trustworthy:
// this package's Go anchor against the LIVE chess-cli C engine, over real
// corpus FENs, requiring EXACT agreement.
//
// It is a test rather than a script so `go test ./...` cannot silently skip it
// on a machine that has the toolchain. Where the toolchain or the engine
// sources are absent it skips loudly with the reason -- a skipped gate that
// reads as green is the trap README.md:140-143 already warns about, so
// anchor_parity.ps1 treats a SKIP as a failure.
func TestAnchorParity(t *testing.T) {
	zig := findZig()
	if zig == "" {
		t.Skip("zig not found (PATH or D:\\env\\zig) - cannot build the C probe")
	}
	cPort := findCPort(t)
	if cPort == "" {
		t.Skip("chess-cli/c_port not found - cannot build the C probe")
	}

	fens := sampleFENs(t, 20000)
	if len(fens) == 0 {
		t.Skip("no chess corpus found - nothing to compare over")
	}
	t.Logf("comparing %d FENs against %s", len(fens), cPort)

	probe := buildProbe(t, zig, cPort)

	cOut := runProbe(t, probe, fens)
	if len(cOut) != len(fens) {
		t.Fatalf("probe returned %d lines for %d FENs", len(cOut), len(fens))
	}

	var mismatch, bad int
	for i, fen := range fens {
		var pos Position
		ok := pos.FromFEN(fen)

		if cOut[i] == "bad" {
			// The engine rejected it; so must we. Disagreeing about legality
			// is a parity failure in its own right.
			if ok {
				bad++
				if bad <= 3 {
					t.Errorf("legality mismatch: C rejects, Go accepts:\n  %s", fen)
				}
			}
			continue
		}
		if !ok {
			bad++
			if bad <= 3 {
				t.Errorf("legality mismatch: C accepts, Go rejects:\n  %s", fen)
			}
			continue
		}
		want, err := strconv.Atoi(cOut[i])
		if err != nil {
			t.Fatalf("probe line %d not an integer: %q", i, cOut[i])
		}
		if got := pos.EvalClassical(); got != want {
			mismatch++
			if mismatch <= 5 {
				t.Errorf("anchor mismatch: go=%d c=%d\n  %s", got, want, fen)
			}
		}
	}
	if mismatch != 0 || bad != 0 {
		t.Fatalf("ANCHOR PARITY FAIL: %d eval mismatches, %d legality mismatches over %d FENs",
			mismatch, bad, len(fens))
	}
	t.Logf("ANCHOR PARITY PASS: %d FENs, 0 mismatches", len(fens))
}

// TestFENRoundTrip checks FromFEN -> ToFEN reproduces the input. The corpus is
// written by pos_to_fen, so a faithful parser must round-trip it byte for byte;
// a field silently dropped on the way in would otherwise stay invisible until
// the dedup key started merging distinct positions.
func TestFENRoundTrip(t *testing.T) {
	fens := sampleFENs(t, 50000)
	if len(fens) == 0 {
		t.Skip("no chess corpus found")
	}
	var bad int
	for _, fen := range fens {
		var pos Position
		if !pos.FromFEN(fen) {
			t.Errorf("failed to parse: %s", fen)
			bad++
			if bad > 5 {
				t.Fatal("too many parse failures")
			}
			continue
		}
		if got := pos.ToFEN(); got != fen {
			t.Errorf("round-trip differs:\n  in  %s\n  out %s", fen, got)
			bad++
			if bad > 5 {
				t.Fatal("too many round-trip failures")
			}
		}
	}
	t.Logf("round-tripped %d FENs", len(fens))
}

// TestFeatureExtraction checks the invariants the corpus loader relies on.
func TestFeatureExtraction(t *testing.T) {
	var pos Position
	if !pos.FromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") {
		t.Fatal("start position failed to parse")
	}
	feats := pos.AppendFeatures(nil)
	if len(feats) != 32 {
		t.Fatalf("start position has %d features, want 32", len(feats))
	}
	for _, f := range feats {
		if int(f) >= NumFeatures {
			t.Fatalf("feature %d out of range (max %d)", f, NumFeatures-1)
		}
	}

	// Color symmetry: the start position mirrored and with the side to move
	// swapped must produce the SAME feature multiset, because features are
	// stm-relative. This is the property that makes `chess evalsym` pass.
	var mirrored Position
	if !mirrored.FromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1") {
		t.Fatal("mirrored start failed to parse")
	}
	mf := mirrored.AppendFeatures(nil)
	if !sameMultiset(feats, mf) {
		t.Error("stm-relative features are not symmetric on the start position")
	}
}

func sameMultiset(a, b []uint16) bool {
	if len(a) != len(b) {
		return false
	}
	m := map[uint16]int{}
	for _, x := range a {
		m[x]++
	}
	for _, x := range b {
		m[x]--
		if m[x] < 0 {
			return false
		}
	}
	return true
}

// ── helpers ────────────────────────────────────────────────────────────────

func findZig() string {
	if p, err := exec.LookPath("zig"); err == nil {
		return p
	}
	for _, c := range []string{`D:\env\zig\zig.exe`, `/usr/local/bin/zig`} {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return ""
}

// workspaceRoot walks up from this source file to the directory holding the
// sibling repos. Using runtime.Caller rather than the CWD keeps the test
// working under `go test ./...` from any directory.
func workspaceRoot() string {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return ""
	}
	// .../stand-alone-trainer/internal/chess/parity_test.go
	return filepath.Dir(filepath.Dir(filepath.Dir(filepath.Dir(file))))
}

func findCPort(t *testing.T) string {
	t.Helper()
	p := filepath.Join(workspaceRoot(), "chess-cli", "c_port")
	if _, err := os.Stat(filepath.Join(p, "eval.c")); err != nil {
		return ""
	}
	return p
}

// chessDataDirs are every place a chess corpus may sit, in preference order.
func chessDataDirs() []string {
	ws := workspaceRoot()
	return []string{
		filepath.Join(ws, "stand-alone-trainer", "data", "chess"),
		filepath.Join(ws, "stand-alone-trainer", "data", "chess960"),
		filepath.Join(ws, "chess-cli", "c_port", "tools", "trainer", "data"),
	}
}

// sampleFENs pulls up to n FENs out of the real corpus. Lines are
// "<fen> <result> <gameid> [<eval>]" with the FEN spanning the first 6
// whitespace fields.
//
// It spreads the sample across every corpus file and STRIDES within each one.
// Taking the first n lines of one file would draw almost entirely from the
// openings of a handful of games, which is the region where the eval terms
// least differ -- an anchor bug in a rook endgame would sail through. The
// stride is deliberately coprime-ish with nothing in particular; it just has
// to walk the whole file.
func sampleFENs(t *testing.T, n int) []string {
	t.Helper()
	var files []string
	for _, dir := range chessDataDirs() {
		m, _ := filepath.Glob(filepath.Join(dir, "*.txt"))
		files = append(files, m...)
	}
	if len(files) == 0 {
		return nil
	}

	perFile := n/len(files) + 1
	out := make([]string, 0, n)
	for _, f := range files {
		if len(out) >= n {
			break
		}
		fh, err := os.Open(f)
		if err != nil {
			continue
		}
		// Stride so the sample reaches deep into the file (late middlegame and
		// endgame positions) instead of stopping in the openings.
		stride := 1
		if fi, err := fh.Stat(); err == nil && perFile > 0 {
			approxLines := fi.Size() / 90 // corpus lines average ~90 bytes
			if s := int(approxLines) / perFile; s > 1 {
				stride = s
			}
		}
		sc := bufio.NewScanner(fh)
		sc.Buffer(make([]byte, 0, 1<<20), 1<<20)
		taken, i := 0, 0
		for sc.Scan() && taken < perFile && len(out) < n {
			if i%stride == 0 {
				if fen, ok := fenFromLine(sc.Text()); ok {
					out = append(out, fen)
					taken++
				}
			}
			i++
		}
		fh.Close()
	}
	return out
}

// fenFromLine rejoins the first six fields of a corpus line into a FEN.
func fenFromLine(line string) (string, bool) {
	f := strings.Fields(line)
	if len(f) < 6 {
		return "", false
	}
	return strings.Join(f[:6], " "), true
}

func buildProbe(t *testing.T, zig, cPort string) string {
	t.Helper()
	dir := t.TempDir()
	exe := filepath.Join(dir, "probe_eval")
	if runtime.GOOS == "windows" {
		exe += ".exe"
	}
	src := filepath.Join(workspaceRoot(), "stand-alone-trainer",
		"internal", "chess", "testdata", "probe_eval.c")

	args := []string{"cc", "-O2", "-std=c11", "-mcpu=baseline", "-I.",
		"-o", exe, src, "bitboard.c", "position.c", "eval.c"}
	cmd := exec.Command(zig, args...)
	cmd.Dir = cPort
	// zig's global cache chokes on spaced paths; the repo's own build.bat sets
	// the same override.
	cmd.Env = append(os.Environ(), "ZIG_GLOBAL_CACHE_DIR=D:\\zcache")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("building the C probe failed: %v\n%s", err, out)
	}
	return exe
}

func runProbe(t *testing.T, exe string, fens []string) []string {
	t.Helper()
	var in bytes.Buffer
	for _, f := range fens {
		in.WriteString(f)
		in.WriteByte('\n')
	}
	cmd := exec.Command(exe)
	cmd.Stdin = &in
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("running the C probe failed: %v", err)
	}
	lines := strings.Split(strings.ReplaceAll(string(out), "\r\n", "\n"), "\n")
	for len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	return lines
}
