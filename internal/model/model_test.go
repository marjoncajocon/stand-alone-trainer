package model_test

import (
	"math/rand"
	"testing"

	"nnuetrainer/internal/emit"
	"nnuetrainer/internal/model"
	"nnuetrainer/internal/variant"
)

// draughtsNames is variant.Names() minus the chess kernel.
//
// Every test below asserts a property of the DRAUGHTS kernel specifically --
// the mirror permutation, f(b) = g(b) - g(mirror(b)) antisymmetry, the int16
// narrow accumulator, the NTW1 layout, phase buckets. Chess has none of them:
// its symmetry lives in the input encoding, it has one accumulator, its C
// kernel accumulates in int32 by design (position.h:34-42), and it uses NTW2.
// Filtering here rather than skipping inside each test keeps the reason in one
// place. Chess's equivalents live in internal/chess and in TestChessKernel.
func draughtsNames() []string {
	var out []string
	for _, n := range variant.Names() {
		v, err := variant.Get(n)
		if err != nil || v.IsChess() {
			continue
		}
		out = append(out, n)
	}
	return out
}

// mirrorBoard applies the 180-degree rotation plus colour swap that the network
// architecture is built around. On every variant here the board string is
// row-major over Board*Board cells, so rotating 180 degrees is exactly
// reversing the string; the colour swap is X<->O and K<->Q.
func mirrorBoard(b string) string {
	out := make([]byte, len(b))
	for i := 0; i < len(b); i++ {
		switch b[len(b)-1-i] {
		case 'X':
			out[i] = 'O'
		case 'O':
			out[i] = 'X'
		case 'K':
			out[i] = 'Q'
		case 'Q':
			out[i] = 'K'
		default:
			out[i] = '.'
		}
	}
	return string(out)
}

func randomBoard(v *variant.Variant, rng *rand.Rand) string {
	b := make([]byte, v.Cells)
	for i := range b {
		b[i] = '.'
	}
	glyphs := []byte{'X', 'O', 'K', 'Q'}
	n := 6 + rng.Intn(20)
	for k := 0; k < n; k++ {
		r, c := rng.Intn(v.Board), rng.Intn(v.Board)
		if v.SquareOf(r, c) < 0 {
			continue
		}
		b[r*v.Board+c] = glyphs[rng.Intn(len(glyphs))]
	}
	return string(b)
}

func randQuant(v *variant.Variant, h int, rng *rand.Rand) *model.Quant {
	q := &model.Quant{
		H: h, NumFeat: v.NumFeat, Buckets: v.Buckets, Mirror: v.Mirror(),
		W1: make([]int16, v.NumFeat*h),
		B1: make([]int32, h),
		W2: make([]int16, v.Buckets*h),
	}
	for i := range q.W1 {
		q.W1[i] = int16(rng.Intn(400) - 200)
	}
	for i := range q.B1 {
		q.B1[i] = int32(rng.Intn(60) - 10)
	}
	for i := range q.W2 {
		q.W2[i] = int16(rng.Intn(400) - 200)
	}
	return q
}

// TestMirrorIsInvolution: the mirror permutation must be its own inverse, or
// f(b) = g(b) - g(mirror(b)) is not antisymmetric.
func TestMirrorIsInvolution(t *testing.T) {
	for _, name := range draughtsNames() {
		v, err := variant.Get(name)
		if err != nil {
			t.Fatal(err)
		}
		m := v.Mirror()
		seen := make([]bool, v.NumFeat)
		for f := 0; f < v.NumFeat; f++ {
			mf := int(m[f])
			if mf < 0 || mf >= v.NumFeat {
				t.Fatalf("%s: mirror[%d] = %d out of range", name, f, mf)
			}
			if int(m[mf]) != f {
				t.Errorf("%s: mirror is not an involution at %d (-> %d -> %d)", name, f, mf, m[mf])
			}
			if seen[mf] {
				t.Errorf("%s: mirror is not a permutation, %d hit twice", name, mf)
			}
			seen[mf] = true
		}
	}
}

// TestAntisymmetry is the property the entire design rests on: swapping colours
// and rotating the board must negate the score EXACTLY, including after
// quantization. The old hand eval violated this on 86.1% of positions and the
// search overvalued grabbing material as a result.
func TestAntisymmetry(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for _, name := range draughtsNames() {
		v, _ := variant.Get(name)
		for _, h := range []int{32, 64} {
			q := randQuant(v, h, rng)
			for trial := 0; trial < 200; trial++ {
				b := randomBoard(v, rng)
				mb := mirrorBoard(b)

				fa, anchorA, piecesA := v.AppendFeats(nil, b)
				fb, anchorB, piecesB := v.AppendFeats(nil, mb)

				if piecesA != piecesB {
					t.Fatalf("%s: mirror changed the piece count (%d vs %d)", name, piecesA, piecesB)
				}
				if anchorA != -anchorB {
					t.Fatalf("%s: anchor not antisymmetric (%d vs %d)", name, anchorA, anchorB)
				}
				// Piece count is mirror-invariant, so both sides select the same
				// W2 row - that is what lets a bucketed net stay antisymmetric.
				if v.Bucket(piecesA) != v.Bucket(piecesB) {
					t.Fatalf("%s: bucket not mirror-invariant", name)
				}

				sa := q.Score16(fa, v.Bucket(piecesA))
				sb := q.Score16(fb, v.Bucket(piecesB))
				if sa != -sb {
					t.Fatalf("%s H=%d: score16 not antisymmetric: %d vs %d\n  b=%s\n  m=%s",
						name, h, sa, sb, b, mb)
				}
			}
		}
	}
}

// TestNarrowKernelAgrees checks the int16 accumulator the engine actually uses
// against the int32 reference, on nets small enough that no overflow is possible.
func TestNarrowKernelAgrees(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for _, name := range draughtsNames() {
		v, _ := variant.Get(name)
		q := randQuant(v, 64, rng)
		for trial := 0; trial < 200; trial++ {
			b := randomBoard(v, rng)
			f, _, pieces := v.AppendFeats(nil, b)
			bk := v.Bucket(pieces)
			if got, want := q.Score16Narrow(f, bk), q.Score16(f, bk); got != want {
				t.Fatalf("%s: int16 kernel %d != int32 reference %d on %s", name, got, want, b)
			}
		}
	}
}

// TestHeaderRoundTrip: what we write must read back identically, and the
// #define block must reflect the architecture.
func TestBinRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for _, name := range draughtsNames() {
		v, _ := variant.Get(name)
		q := randQuant(v, 32, rng)
		p := t.TempDir() + "/nn.ntw"
		if err := emit.Bin(p, q, v); err != nil {
			t.Fatal(err)
		}
		got, err := emit.LoadBin(p, v)
		if err != nil {
			t.Fatal(err)
		}
		if got.H != q.H || got.NumFeat != q.NumFeat || got.Buckets != q.Buckets {
			t.Fatalf("%s: shape drift %v/%v/%v", name, got.H, got.NumFeat, got.Buckets)
		}
		for i := range q.W1 {
			if got.W1[i] != q.W1[i] {
				t.Fatalf("%s: W1[%d] drift", name, i)
			}
		}
		for i := range q.B1 {
			if got.B1[i] != q.B1[i] {
				t.Fatalf("%s: B1[%d] drift", name, i)
			}
		}
		for i := range q.W2 {
			if got.W2[i] != q.W2[i] {
				t.Fatalf("%s: W2[%d] drift", name, i)
			}
		}
	}
}

// TestBucketThresholds pins the phase-bucket boundaries, which must match the
// SAME function in each engine's search.c and probe_nnue.c. Three sites, one
// contract; drift here silently changes which weights the engine applies.
func TestBucketThresholds(t *testing.T) {
	fil, _ := variant.Get("filipino")
	for _, c := range []struct{ pieces, want int }{
		{24, 0}, {20, 0}, {19, 1}, {14, 1}, {13, 2}, {8, 2}, {7, 3}, {6, 3},
	} {
		if got := fil.Bucket(c.pieces); got != c.want {
			t.Errorf("filipino Bucket(%d) = %d, want %d", c.pieces, got, c.want)
		}
	}
	tur, _ := variant.Get("turkish")
	for _, c := range []struct{ pieces, want int }{
		{32, 0}, {26, 0}, {25, 1}, {18, 1}, {17, 2}, {10, 2}, {9, 3},
	} {
		if got := tur.Bucket(c.pieces); got != c.want {
			t.Errorf("turkish Bucket(%d) = %d, want %d", c.pieces, got, c.want)
		}
	}
	// Single-bucket variants must fold everything to row 0.
	eng, _ := variant.Get("english")
	for _, p := range []int{2, 8, 14, 20, 24} {
		if got := eng.Bucket(p); got != 0 {
			t.Errorf("english Bucket(%d) = %d, want 0", p, got)
		}
	}
}
