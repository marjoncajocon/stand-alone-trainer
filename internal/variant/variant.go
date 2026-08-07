// Package variant holds one descriptor per checkers/draughts variant.
//
// Everything that differs between the six engines lives here and nowhere else:
// board geometry, the playable-square numbering, material anchor values, the
// number of W2 phase buckets the engine's kernel actually implements, the
// phase-bucket thresholds, and the tablebase-territory skip rule.
//
// Every constant below was read out of that repo's own
// c_port/tools/trainer/trainer/nnue/train_nnue.go, NOT guessed. Where a value
// looks surprising (english kings are worth 150, brazilian skips nothing but
// empty boards) it is surprising in the source too, and a comment says so.
//
// The one thing a descriptor may NOT change is the network topology. Every
// engine's bb_nnue_score16 implements exactly one hidden layer with a
// per-bucket output row, so the trainer produces exactly that and nothing else.
package variant

import "fmt"

// SkipFunc reports whether a position is tablebase territory and must be
// dropped: the engine answers those from its TB probe before eval ever runs,
// so training on them teaches the net positions it will never be asked about.
type SkipFunc func(xMen, oMen, xKings, oKings int) bool

// Variant is a complete description of one engine's NNUE input contract.
type Variant struct {
	Name      string // --variant value, and the data-file infix
	EngineDir string // where nnue_weights.h gets installed, relative to the workspace

	Board   int // 8 or 10
	Cells   int // Board*Board — the board string length in the data
	Squares int // playable squares: 32 diagonal, 50 diagonal (10x10), 64 all
	NumFeat int // 4 * Squares

	ValMan  int // frozen material anchor, engine units
	ValKing int

	// Buckets is what that repo's search.c implements TODAY. english, russian
	// and international have no NN_W2_BUCKETS branch at all, so they get 1 and
	// the generated header omits the #define entirely. Raising it there would
	// emit a header their engines cannot compile.
	Buckets   int
	PhaseCuts []int // len == Buckets-1, descending piece-count thresholds

	// AllSquares is true only for turkish, where men move orthogonally and
	// every one of the 64 squares is playable.
	AllSquares bool

	TBSkip  SkipFunc
	SkipDoc string // human-readable form of the rule, printed at load time
	FeatDoc string // goes verbatim into the generated header comment
}

var registry = map[string]*Variant{}

func register(v *Variant) {
	if err := v.Validate(); err != nil {
		panic(fmt.Sprintf("variant %s: %v", v.Name, err))
	}
	registry[v.Name] = v
}

// Get returns the descriptor for name, or an error listing the known variants.
func Get(name string) (*Variant, error) {
	if v, ok := registry[name]; ok {
		return v, nil
	}
	return nil, fmt.Errorf("unknown variant %q (known: %v)", name, Names())
}

// Names returns every registered variant, in a stable order.
func Names() []string {
	return []string{"filipino", "brazilian", "english", "russian", "international", "turkish"}
}

// Validate catches a descriptor that contradicts itself before any data is read.
func (v *Variant) Validate() error {
	if v.Cells != v.Board*v.Board {
		return fmt.Errorf("Cells %d != Board^2 %d", v.Cells, v.Board*v.Board)
	}
	if v.NumFeat != 4*v.Squares {
		return fmt.Errorf("NumFeat %d != 4*Squares %d", v.NumFeat, 4*v.Squares)
	}
	if v.NumFeat > 65535 {
		return fmt.Errorf("NumFeat %d exceeds the uint16 feature index", v.NumFeat)
	}
	if v.Buckets < 1 {
		return fmt.Errorf("Buckets %d < 1", v.Buckets)
	}
	if len(v.PhaseCuts) != v.Buckets-1 {
		return fmt.Errorf("PhaseCuts has %d entries, want Buckets-1 = %d",
			len(v.PhaseCuts), v.Buckets-1)
	}
	for i := 1; i < len(v.PhaseCuts); i++ {
		if v.PhaseCuts[i] >= v.PhaseCuts[i-1] {
			return fmt.Errorf("PhaseCuts must be strictly descending, got %v", v.PhaseCuts)
		}
	}
	if v.TBSkip == nil {
		return fmt.Errorf("TBSkip is nil (use NoSkip for variants that skip nothing)")
	}
	return nil
}

// SquareOf maps a board coordinate to its playable-square index, or -1.
func (v *Variant) SquareOf(r, c int) int {
	if v.AllSquares {
		return r*v.Board + c
	}
	if (r+c)%2 != 1 {
		return -1 // pieces only live on dark squares
	}
	return (v.Board/2)*r + c/2
}

// Bucket maps a total piece count to its W2 output row. Piece count is
// MIRROR-INVARIANT, which is what lets both accumulators share one W2 and keeps
// f(b) = g(b) - g(mirror(b)) structurally antisymmetric after quantization.
func (v *Variant) Bucket(pieces int) int {
	for i, cut := range v.PhaseCuts {
		if pieces >= cut {
			return i
		}
	}
	return len(v.PhaseCuts)
}

// Mirror builds the feature permutation for a 180-degree rotation plus colour
// swap: plane^1, and square s -> Squares-1-s. Uniform across all six variants.
func (v *Variant) Mirror() []uint16 {
	m := make([]uint16, v.NumFeat)
	for f := 0; f < v.NumFeat; f++ {
		p, s := f/v.Squares, f%v.Squares
		m[f] = uint16((p^1)*v.Squares + (v.Squares - 1 - s))
	}
	return m
}

// Counts tallies the four piece kinds on a board string.
func (v *Variant) Counts(b string) (xMen, oMen, xKings, oKings int) {
	for i := 0; i < v.Cells; i++ {
		switch b[i] {
		case 'X':
			xMen++
		case 'O':
			oMen++
		case 'K':
			xKings++
		case 'Q':
			oKings++
		}
	}
	return
}

// AppendFeats appends a board's active feature indices to dst (a shared pool,
// so no per-board allocation) and returns the frozen material anchor and the
// piece count.
//
// Iteration order is row-major and the plane order is X, O, K, Q — identical to
// every reference trainer, because the feature list order feeds straight into
// the training loop's accumulate order.
func (v *Variant) AppendFeats(dst []uint16, b string) (out []uint16, anchor, pieces int) {
	out = dst
	sq := v.Squares
	for r := 0; r < v.Board; r++ {
		for c := 0; c < v.Board; c++ {
			ch := b[r*v.Board+c]
			if ch == '.' {
				continue
			}
			s := v.SquareOf(r, c)
			if s < 0 {
				continue
			}
			switch ch {
			case 'X':
				out = append(out, uint16(s))
				anchor += v.ValMan
			case 'O':
				out = append(out, uint16(sq+s))
				anchor -= v.ValMan
			case 'K':
				out = append(out, uint16(2*sq+s))
				anchor += v.ValKing
			case 'Q':
				out = append(out, uint16(3*sq+s))
				anchor -= v.ValKing
			default:
				continue // unknown glyph: not a piece, not counted
			}
			pieces++
		}
	}
	return
}

// NoSkip is the TBSkip for variants whose trainer drops nothing.
func NoSkip(int, int, int, int) bool { return false }

// skipTotalAtMost builds a "drop when total pieces <= n" rule.
func skipTotalAtMost(n int) SkipFunc {
	return func(xm, om, xk, ok int) bool { return xm+om+xk+ok <= n }
}

func init() {
	// ── filipino — the reference variant, and the only one with a live corpus ──
	register(&Variant{
		Name: "filipino", EngineDir: "dama-cli/c_port",
		Board: 8, Cells: 64, Squares: 32, NumFeat: 128,
		ValMan: 100, ValKing: 320,
		Buckets: 4, PhaseCuts: []int{20, 14, 8},
		TBSkip:  skipTotalAtMost(5),
		SkipDoc: "total pieces <= 5 (engine answers those from the tablebase)",
		FeatDoc: "feat = plane*32 + (4r + c/2); planes: 0 X-man, 1 O-man, 2 X-king, 3 O-king",
	})

	// ── brazilian — identical geometry to filipino; note it skips almost nothing.
	// Its trainer drops only pieces == 0, i.e. empty boards, NOT tablebase
	// territory. Copied deliberately: changing it would silently retrain on a
	// different position set than every brazilian net shipped so far.
	register(&Variant{
		Name: "brazilian", EngineDir: "brazilian-cli/c_port",
		Board: 8, Cells: 64, Squares: 32, NumFeat: 128,
		ValMan: 100, ValKing: 320,
		Buckets: 4, PhaseCuts: []int{20, 14, 8},
		TBSkip:  skipTotalAtMost(0),
		SkipDoc: "empty boards only (its trainer has no tablebase skip)",
		FeatDoc: "feat = plane*32 + (4r + c/2); planes: 0 X-man, 1 O-man, 2 X-king, 3 O-king",
	})

	// ── english / american checkers — kings are worth 150, not 320, and the
	// skip is a 1-vs-3 endgame rule rather than a piece total. Its search.c has
	// NO NN_W2_BUCKETS branch, so Buckets must stay 1.
	register(&Variant{
		Name: "english", EngineDir: "english-dama-cli/c_port",
		Board: 8, Cells: 64, Squares: 32, NumFeat: 128,
		ValMan: 100, ValKing: 150,
		Buckets: 1, PhaseCuts: nil,
		TBSkip: func(xm, om, xk, ok int) bool {
			x, o := xm+xk, om+ok
			weak, strong := x, o
			if o < x {
				weak, strong = o, x
			}
			return weak <= 1 && strong <= 3
		},
		SkipDoc: "weaker side <= 1 piece and stronger side <= 3",
		FeatDoc: "feat = plane*32 + (4r + c/2); planes: 0 X-man, 1 O-man, 2 X-king, 3 O-king",
	})

	// ── russian / shashki — filipino geometry and material, no skip, single W2.
	register(&Variant{
		Name: "russian", EngineDir: "rusian-dama-cli/c_port", // folder really is "rusian"
		Board: 8, Cells: 64, Squares: 32, NumFeat: 128,
		ValMan: 100, ValKing: 320,
		Buckets: 1, PhaseCuts: nil,
		TBSkip:  NoSkip,
		SkipDoc: "none",
		FeatDoc: "feat = plane*32 + (4r + c/2); planes: 0 X-man, 1 O-man, 2 X-king, 3 O-king",
	})

	// ── international / FMJD — the only 10x10 board: 100-char board strings,
	// 50 dark squares, 200 features. Its shipped net is H=32.
	register(&Variant{
		Name: "international", EngineDir: "international-dama-cli/c_port",
		Board: 10, Cells: 100, Squares: 50, NumFeat: 200,
		ValMan: 100, ValKing: 320,
		Buckets: 1, PhaseCuts: nil,
		TBSkip:  NoSkip,
		SkipDoc: "none",
		FeatDoc: "feat = plane*50 + (5r + c/2); planes: 0 X-man, 1 O-man, 2 X-king, 3 O-king",
	})

	// ── turkish — men move orthogonally, so ALL 64 squares are playable and the
	// feature space is 256, not 128. Phase cuts scale with the 32-piece start.
	register(&Variant{
		Name: "turkish", EngineDir: "turkish-cli/c_port",
		Board: 8, Cells: 64, Squares: 64, NumFeat: 256,
		ValMan: 100, ValKing: 320,
		Buckets: 4, PhaseCuts: []int{26, 18, 10},
		AllSquares: true,
		TBSkip:     skipTotalAtMost(2),
		SkipDoc:    "total pieces <= 2 (3-piece tablebase territory)",
		FeatDoc:    "feat = plane*64 + (8r + c); planes: 0 X-man, 1 O-man, 2 X-king, 3 O-king",
	})
}
