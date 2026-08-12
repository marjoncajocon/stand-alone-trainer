package emit_test

import (
	"bytes"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"nnuetrainer/internal/chess"
	"nnuetrainer/internal/emit"
	"nnuetrainer/internal/model"
	"nnuetrainer/internal/variant"
)

// TestChessKernelParity is the gate that makes "full replacement" a claim
// rather than a hope: our integer reference (model.Quant.ScoreCP) against
// chess-cli's OWN nnue.c, compiled fresh against the header WE generate.
//
// Compiling the probe from the live engine source is the point. A prebuilt
// probe measures whatever header sat next to it that day -- the trap
// README.md:140-143 records for probe_nnue.exe and parity.ps1:10-14 exists to
// avoid.
func TestChessKernelParity(t *testing.T) {
	zig, cPort := requireToolchain(t)
	v, err := variant.Get("chess")
	if err != nil {
		t.Fatal(err)
	}

	m, q := randomChessNet(v, 4242)
	dir := t.TempDir()

	hdr := filepath.Join(dir, "nnue_weights.h")
	if err := emit.Header(hdr, q, v); err != nil {
		t.Fatal(err)
	}

	fens := sampleFENs(t, 400)
	if len(fens) == 0 {
		t.Skip("no chess corpus found to sample FENs from")
	}

	probe := buildChessProbe(t, zig, cPort, dir)
	cOut := runLines(t, probe, fens)
	if len(cOut) != len(fens) {
		t.Fatalf("probe returned %d lines for %d FENs", len(cOut), len(fens))
	}

	var mismatch int
	for i, fen := range fens {
		var pos chess.Position
		if !pos.FromFEN(fen) {
			t.Fatalf("Go rejected a FEN the corpus contains: %s", fen)
		}
		got := q.ScoreCP(pos.AppendFeatures(nil))
		want, err := strconv.Atoi(cOut[i])
		if err != nil {
			t.Fatalf("probe line %d not an integer: %q", i, cOut[i])
		}
		if got != want {
			mismatch++
			if mismatch <= 5 {
				t.Errorf("kernel mismatch: go=%d c=%d\n  %s", got, want, fen)
			}
		}
	}
	if mismatch != 0 {
		t.Fatalf("CHESS KERNEL PARITY FAIL: %d/%d mismatches", mismatch, len(fens))
	}
	t.Logf("CHESS KERNEL PARITY PASS: %d FENs, 0 mismatches", len(fens))
	_ = m
}

// TestChessHeaderMatchesN2H is the second, independent check of the
// quantization contract: our float weights -> net.bin -> chess-cli's OWN
// n2h.exe -> a header, which must be BYTE-IDENTICAL to the one we write
// directly.
//
// One diff validates every scale at once (QA on w1/b1, QB on w2, QA*QB on b2)
// against the tool that has been producing the shipped headers all along. If
// this passes, n2h.c and trainer.c are genuinely redundant.
func TestChessHeaderMatchesN2H(t *testing.T) {
	zig, cPort := requireToolchain(t)
	v, _ := variant.Get("chess")

	m, q := randomChessNet(v, 99)
	dir := t.TempDir()

	ours := filepath.Join(dir, "ours.h")
	if err := emit.Header(ours, q, v); err != nil {
		t.Fatal(err)
	}
	netBin := filepath.Join(dir, "net.bin")
	if err := emit.NetBin(netBin, m); err != nil {
		t.Fatal(err)
	}

	n2h := filepath.Join(dir, "n2h")
	if runtime.GOOS == "windows" {
		n2h += ".exe"
	}
	build := exec.Command(zig, "cc", "-O2", "-std=c11", "-o", n2h,
		filepath.Join(cPort, "tools", "n2h.c"))
	build.Env = append(os.Environ(), "ZIG_GLOBAL_CACHE_DIR=D:\\zcache")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("building n2h failed: %v\n%s", err, out)
	}

	theirs := filepath.Join(dir, "theirs.h")
	run := exec.Command(n2h, netBin, theirs)
	if out, err := run.CombinedOutput(); err != nil {
		t.Fatalf("running n2h failed: %v\n%s", err, out)
	}

	a, err := os.ReadFile(ours)
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(theirs)
	if err != nil {
		t.Fatal(err)
	}

	// Two differences are legitimate and neither touches a weight:
	//
	//   - The banner comment names the producing tool and its input file, so
	//     compare from the include guard on.
	//   - n2h.exe opens its output with fopen(..., "w"), which on Windows is
	//     TEXT mode and turns every \n into \r\n. We write LF unconditionally
	//     (emit.go's header contract says so). Normalize before comparing, or
	//     this test would fail on Windows and pass on Linux for the same bytes.
	//
	// Everything else -- every quantized value, in order -- must match.
	ca, cb := normalize(a), normalize(b)
	if !bytes.Equal(ca, cb) {
		t.Errorf("QUANTIZATION MISMATCH vs n2h.exe: %s", firstDiff(ca, cb))
	} else {
		t.Logf("header body byte-identical to n2h.exe (%d bytes)", len(ca))
	}
}

// TestChessNTWRoundTrip: NTW2 must carry B2, and must refuse to load as NTW1.
func TestChessNTWRoundTrip(t *testing.T) {
	v, _ := variant.Get("chess")
	_, q := randomChessNet(v, 7)
	p := filepath.Join(t.TempDir(), "nn.ntw")
	if err := emit.Bin(p, q, v); err != nil {
		t.Fatal(err)
	}
	got, err := emit.LoadBin(p, v)
	if err != nil {
		t.Fatal(err)
	}
	if !got.Chess {
		t.Error("round-tripped net lost its chess kernel flag")
	}
	if got.B2 != q.B2 {
		t.Errorf("B2 drift: %d != %d (NTW1 has no slot for it, which is why "+
			"chess writes NTW2)", got.B2, q.B2)
	}
	if got.H != q.H || got.NumFeat != q.NumFeat || got.Buckets != q.Buckets {
		t.Fatalf("shape drift")
	}
	for i := range q.W1 {
		if got.W1[i] != q.W1[i] {
			t.Fatalf("W1[%d] drift", i)
		}
	}
	for i := range q.W2 {
		if got.W2[i] != q.W2[i] {
			t.Fatalf("W2[%d] drift", i)
		}
	}

	// A chess file must not load under a draughts variant.
	fil, _ := variant.Get("filipino")
	if _, err := emit.LoadBin(p, fil); err == nil {
		t.Error("an NTW2 chess net loaded as filipino; it must not")
	}
}

// ── helpers ────────────────────────────────────────────────────────────────

// randomChessNet builds a float model with weights in a realistic range and its
// quantized form. Values are deliberately large enough that the int32
// accumulator matters -- a small net would hide a narrowing bug.
func randomChessNet(v *variant.Variant, seed int64) (*model.Model, *model.Quant) {
	rng := rand.New(rand.NewSource(seed))
	m := model.New(v, 256, 1, rng)
	for i := range m.W1 {
		m.W1[i] = (rng.Float64()*2 - 1) * 0.5
	}
	for j := range m.B1 {
		m.B1[j] = (rng.Float64()*2 - 1) * 0.3
	}
	for j := range m.W2 {
		m.W2[j] = (rng.Float64()*2 - 1) * 0.8
	}
	m.B2 = (rng.Float64()*2 - 1) * 2.0
	return m, m.Quantize()
}

func requireToolchain(t *testing.T) (zig, cPort string) {
	t.Helper()
	if p, err := exec.LookPath("zig"); err == nil {
		zig = p
	} else if _, err := os.Stat(`D:\env\zig\zig.exe`); err == nil {
		zig = `D:\env\zig\zig.exe`
	} else {
		t.Skip("zig not found - cannot compile the C leg")
	}
	cPort = filepath.Join(workspaceRoot(), "chess-cli", "c_port")
	if _, err := os.Stat(filepath.Join(cPort, "nnue.c")); err != nil {
		t.Skip("chess-cli/c_port not found")
	}
	return zig, cPort
}

func workspaceRoot() string {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return ""
	}
	// .../stand-alone-trainer/internal/emit/chess_parity_test.go
	return filepath.Dir(filepath.Dir(filepath.Dir(filepath.Dir(file))))
}

// buildChessProbe compiles a probe that evaluates nnue_eval against OUR header.
//
// nnue.c is COPIED into hdrDir on purpose. A quoted #include searches the
// including file's own directory first, so a copy left in c_port would pick up
// c_port's installed nnue_weights.h and quietly measure the shipped net instead
// of the one under test.
func buildChessProbe(t *testing.T, zig, cPort, hdrDir string) string {
	t.Helper()
	src, err := os.ReadFile(filepath.Join(cPort, "nnue.c"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(hdrDir, "nnue_copy.c"), src, 0o644); err != nil {
		t.Fatal(err)
	}
	main := `#include <stdio.h>
#include <string.h>
#include "position.h"
#include "nnue.h"
int main(void) {
    position_init();
    char line[512];
    while (fgets(line, sizeof(line), stdin)) {
        size_t n = strlen(line);
        while (n && (line[n-1]=='\n' || line[n-1]=='\r')) line[--n] = 0;
        if (!n) continue;
        Position pos;
        if (!pos_from_fen(&pos, line)) { printf("bad\n"); continue; }
        printf("%d\n", nnue_eval(&pos));
    }
    return 0;
}
`
	if err := os.WriteFile(filepath.Join(hdrDir, "probe_main.c"), []byte(main), 0o644); err != nil {
		t.Fatal(err)
	}

	exe := filepath.Join(hdrDir, "probe_chess")
	if runtime.GOOS == "windows" {
		exe += ".exe"
	}
	cmd := exec.Command(zig, "cc", "-O2", "-std=c11", "-mcpu=baseline",
		"-DUSE_NNUE", "-I"+hdrDir, "-I"+cPort,
		"-o", exe,
		"probe_main.c", "nnue_copy.c",
		filepath.Join(cPort, "bitboard.c"), filepath.Join(cPort, "position.c"))
	cmd.Dir = hdrDir
	cmd.Env = append(os.Environ(), "ZIG_GLOBAL_CACHE_DIR=D:\\zcache")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("building the chess probe failed: %v\n%s", err, out)
	}
	return exe
}

func runLines(t *testing.T, exe string, in []string) []string {
	t.Helper()
	var b bytes.Buffer
	for _, s := range in {
		b.WriteString(s)
		b.WriteByte('\n')
	}
	cmd := exec.Command(exe)
	cmd.Stdin = &b
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("probe failed: %v", err)
	}
	lines := strings.Split(strings.ReplaceAll(string(out), "\r\n", "\n"), "\n")
	for len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	return lines
}

func sampleFENs(t *testing.T, n int) []string {
	t.Helper()
	dirs := []string{
		filepath.Join(workspaceRoot(), "stand-alone-trainer", "data", "chess"),
		filepath.Join(workspaceRoot(), "stand-alone-trainer", "data", "chess960"),
		filepath.Join(workspaceRoot(), "chess-cli", "c_port", "tools", "trainer", "data"),
	}
	var out []string
	for _, d := range dirs {
		files, _ := filepath.Glob(filepath.Join(d, "*.txt"))
		for _, f := range files {
			raw, err := os.Open(f)
			if err != nil {
				continue
			}
			buf := make([]byte, 1<<20)
			nb, _ := raw.Read(buf)
			raw.Close()
			for _, line := range strings.Split(string(buf[:nb]), "\n") {
				if len(out) >= n {
					return out
				}
				fl := strings.Fields(line)
				if len(fl) >= 6 {
					out = append(out, strings.Join(fl[:6], " "))
				}
			}
		}
	}
	return out
}

// normalize strips the banner comment and folds CRLF to LF, leaving exactly
// the text a compiler would act on.
func normalize(b []byte) []byte {
	if i := bytes.Index(b, []byte("#ifndef NNUE_WEIGHTS_H")); i >= 0 {
		b = b[i:]
	}
	return bytes.ReplaceAll(b, []byte("\r\n"), []byte("\n"))
}

func firstDiff(a, b []byte) string {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			lo := i - 60
			if lo < 0 {
				lo = 0
			}
			return "at byte " + strconv.Itoa(i) + ":\n  ours:   ..." +
				string(a[lo:min(i+60, len(a))]) + "\n  n2h:    ..." +
				string(b[lo:min(i+60, len(b))])
		}
	}
	return "lengths differ: ours " + strconv.Itoa(len(a)) + ", n2h " + strconv.Itoa(len(b))
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
