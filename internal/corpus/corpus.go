// Package corpus loads selfplay text data into deduplicated training samples.
//
// This is a faithful port of the reference loader in dama-cli's train_nnue.go,
// generalized over board geometry. Four behaviours are load-bearing and must
// not be "improved":
//
//   - Dedup key is the BOARD ALONE. The net has no side-to-move input, so a
//     board is one representable position regardless of stm. Each unique board
//     becomes ONE sample whose target is the AVERAGE outcome over every line
//     that produced it — a smooth WDL estimate instead of N noisy {0,.5,1}
//     copies.
//   - Holdout groups are whole GAMES, keyed FNV-1a over "file:gameid". Two
//     positions from the same game are correlated, so splitting them across
//     train/valid would make held-out MSE optimistic. gameid restarts at 0 in
//     every worker file, which is why the file qualifier is part of the key.
//   - accs is kept in FIRST-SEEN order. That order determines the training
//     order and therefore the weights, so it is part of the reproducibility
//     contract, not an implementation detail.
//   - Layout is deliberately flat: one struct slice, packed inline map keys,
//     one shared feature pool. A map[string]*acc version of this was measured
//     to stall for minutes in GC on ~23M lines before printing anything.
package corpus

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"

	"nnuetrainer/internal/chess"
	"nnuetrainer/internal/variant"
)

// maxPackBytes is Cells/2 for the largest board (10x10 = 100 cells). Chess's
// exact position key is 37 bytes (chess.PackKeyLen), so it fits here too.
const maxPackBytes = 50

// Sample is one unique board ready for training.
type Sample struct {
	Off, End int32 // slice into the shared feature pool
	Anchor   int16 // frozen material anchor, engine units
	Bucket   uint8 // phase bucket -> selects the W2 output row
	Y        float64
}

// Stats are the loader counters. They are printed in the reference trainer's
// exact wording so a run can be diffed against a historical log.
type Stats struct {
	Files    int
	Lines    int64
	Unique   int64
	Train    int64
	Valid    int64
	TBSkip   int64
	EvalPos  int64
	Lambda   float64
	KScale   float64
	Holdout  int
	GroupKey string
}

// Set is a loaded corpus.
type Set struct {
	V     *variant.Variant
	Pool  []uint16
	Train []Sample
	Valid []Sample
	Stats Stats
}

// Options control loading.
type Options struct {
	Globs    []string
	Holdout  int
	Lambda   float64
	KScale   float64
	MaxLines int
	// GroupKey selects what the holdout hash is computed over. The split is a
	// function of this string, so changing it changes which games are held out
	// and makes MSE incomparable with a run that used a different setting.
	//
	//	"rel"      <variant>/<basename>   DEFAULT. Stable no matter where the
	//	                                  kit is installed or which OS runs it.
	//	"path"     the glob-matched path  What the reference trainer hashes.
	//	                                  Use only to reproduce a historical run:
	//	                                  "data\x.txt" and "data/x.txt" hash
	//	                                  differently, so this is NOT portable.
	//	"basename" <basename>             Legacy fallback.
	GroupKey string
	Quiet    bool
}

func fnv1aInit() uint32 { return 2166136261 }

func fnv1aBytes(h uint32, b []byte) uint32 {
	for i := 0; i < len(b); i++ {
		h = (h ^ uint32(b[i])) * 16777619
	}
	return h
}

func fnv1aByte(h uint32, b byte) uint32 { return (h ^ uint32(b)) * 16777619 }

// packNib mirrors the reference alphabet: everything outside ".XOKQ" packs to
// 15, so only malformed lines could ever alias.
var packNib [256]uint8

func init() {
	for i := range packNib {
		packNib[i] = 15
	}
	packNib['.'], packNib['X'], packNib['O'], packNib['K'], packNib['Q'] = 0, 1, 2, 3, 4
}

func packKey(b []byte, cells int) (k [maxPackBytes]byte) {
	n := cells / 2
	for i := 0; i < n; i++ {
		k[i] = packNib[b[2*i]]<<4 | packNib[b[2*i+1]]
	}
	return
}

// fenSpan returns the byte length of the first six whitespace-separated fields
// of line -- the FEN -- or -1 if there are fewer than six. Slicing the line is
// what lets the loader parse a FEN without allocating.
func fenSpan(line []byte) int {
	i := 0
	for tok := 0; tok < 6; tok++ {
		for i < len(line) && (line[i] == ' ' || line[i] == '\t' || line[i] == '\r') {
			i++
		}
		if i >= len(line) {
			return -1
		}
		for i < len(line) && line[i] != ' ' && line[i] != '\t' && line[i] != '\r' {
			i++
		}
	}
	return i
}

// chessOutcome maps a self-play result token to WHITE's outcome. Returns false
// for anything else, so a malformed line is skipped rather than silently
// labelled a draw (corpus.go's draughts path defaults unknown glyphs to 0.5,
// which is right there because the alphabet is a single char and a typo cannot
// look like a valid result).
func chessOutcome(tok []byte) (float64, bool) {
	switch string(tok) {
	case "1-0":
		return 1, true
	case "0-1":
		return 0, true
	case "1/2-1/2":
		return 0.5, true
	}
	return 0, false
}

// draughtsAccept is the ONE definition of a well-formed draughts line. Load
// gates on it immediately before nLines++, and count.go's Count calls the same
// function, so an inventory can never disagree with what training reads.
//
// A wrong-length board is the common real failure: it means --variant names a
// different geometry than the data (64 vs 100 cells), and rejecting the line is
// what makes that visible as "everything malformed" rather than as a net that
// trains on nothing.
func draughtsAccept(parts [][]byte, cells int) bool {
	return len(parts) >= 3 && len(parts[0]) == cells
}

// chessAccept is the same contract for a chess line: it parses the FEN into pos
// and returns WHITE's outcome. Two callers, one definition (see count.go).
//
// pos is written even when the outcome token turns out to be bad; callers only
// read it when ok is true.
func chessAccept(line []byte, parts [][]byte, pos *chess.Position) (float64, bool) {
	if len(parts) < 8 {
		return 0, false
	}
	// The FEN is a contiguous prefix of the line, so slice it out rather than
	// rejoining the six tokens -- at this scale a per-line join would be pure
	// garbage.
	n := fenSpan(line)
	if n < 0 || !pos.FromFENBytes(line[:n]) {
		return 0, false
	}
	return chessOutcome(parts[6])
}

// fields splits on whitespace into at most max slices, without allocating.
// Semantics match strings.Fields for the data this reads.
func fields(line []byte, out [][]byte) [][]byte {
	out = out[:0]
	i := 0
	for i < len(line) && len(out) < cap(out) {
		for i < len(line) && (line[i] == ' ' || line[i] == '\t' || line[i] == '\r') {
			i++
		}
		if i >= len(line) {
			break
		}
		j := i
		for j < len(line) && line[j] != ' ' && line[j] != '\t' && line[j] != '\r' {
			j++
		}
		out = append(out, line[i:j])
		i = j
	}
	return out
}

func atoiBytes(b []byte) (int, bool) {
	if len(b) == 0 {
		return 0, false
	}
	i, neg := 0, false
	if b[0] == '-' || b[0] == '+' {
		neg = b[0] == '-'
		i = 1
		if i >= len(b) {
			return 0, false
		}
	}
	n := 0
	for ; i < len(b); i++ {
		if b[i] < '0' || b[i] > '9' {
			return 0, false
		}
		n = n*10 + int(b[i]-'0')
	}
	if neg {
		n = -n
	}
	return n, true
}

type acc struct {
	featOff, featEnd int32
	anchor           int16
	bucket           uint8
	isValid          bool
	n, nEval         int32
	sumOut, sumEval  float64
}

// Load reads every file matching Opts.Globs and returns the deduplicated set.
func Load(v *variant.Variant, o Options) (*Set, error) {
	if o.Holdout <= 0 {
		o.Holdout = 20
	}
	if o.GroupKey == "" {
		o.GroupKey = "rel"
	}
	switch o.GroupKey {
	case "rel", "path", "basename":
	default:
		return nil, fmt.Errorf("--group-key must be rel, path or basename (got %q)", o.GroupKey)
	}

	var files []string
	for _, g := range o.Globs {
		m, err := filepath.Glob(g)
		if err != nil {
			return nil, fmt.Errorf("glob %q: %w", g, err)
		}
		files = append(files, m...)
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("no files match %v", o.Globs)
	}
	// filepath.Glob already sorts per pattern; sort the union so multiple
	// patterns still produce one deterministic order.
	sort.Strings(files)

	var accs []acc
	boardIdx := make(map[[maxPackBytes]byte]int32)
	var pool []uint16
	var nLines, nTBSkip int64

	if !o.Quiet {
		fmt.Printf("loading %d files...\n", len(files))
	}

	isChess := v.IsChess()
	// Chess lines carry a 6-field FEN before the result, so they need 9 slots
	// (fen[0..5] result gameid eval) where draughts needs 5.
	nFields := 5
	if isChess {
		nFields = 9
	}
	fbuf := make([][]byte, 0, nFields)
	var pos chess.Position // reused across lines; nothing is retained

	for fi, f := range files {
		fh, err := os.Open(f)
		if err != nil {
			return nil, fmt.Errorf("open %s: %w", f, err)
		}
		sc := bufio.NewScanner(fh)
		sc.Buffer(make([]byte, 0, 1<<20), 1<<20)

		// Hash the file qualifier once instead of building "file:gameid" per
		// line. FNV-1a is a streaming hash, so this is bit-identical to hashing
		// the concatenated string, without 12M allocations.
		qual := f
		switch o.GroupKey {
		case "rel":
			// <variant>/<basename>: identical on Windows and Linux, and
			// unaffected by where the kit is installed. The variant prefix
			// keeps two variants' identically-named files from colliding.
			//
			// Chess uses the CONTAINING FOLDER instead of v.Name, because
			// --variant=chess pools data/chess and data/chess960. Both folders
			// are written by the same generator with the same naming scheme, so
			// keying on "chess" for both would let a standard file and a 960
			// file with equal basenames share a group -- and then two different
			// games would land on the same side of the holdout together.
			base := v.Name
			if isChess {
				base = filepath.Base(filepath.Dir(f))
			}
			qual = base + "/" + filepath.Base(f)
		case "basename":
			qual = filepath.Base(f)
		}
		fileHash := fnv1aByte(fnv1aBytes(fnv1aInit(), []byte(qual)), ':')

		for sc.Scan() {
			if o.MaxLines > 0 && nLines >= int64(o.MaxLines) {
				break
			}
			parts := fields(sc.Bytes(), fbuf)

			var key [maxPackBytes]byte
			var outcome float64
			var evalCP int
			var hasEval bool
			var gidTok []byte
			var haveGid bool

			if isChess {
				// <fen(6 fields)> <result> <gameid> [<eval>]
				wr, okRes := chessAccept(sc.Bytes(), parts, &pos)
				if !okRes {
					continue
				}
				nLines++

				// PERSPECTIVE FLIP. The file records WHITE's outcome and a
				// WHITE-perspective eval, but the net's output, the anchor and
				// therefore the target are all STM-relative. This is exactly
				// trainer.c:203 and :206. Getting it wrong has no symptom other
				// than a net that plays badly, so it is asserted directly in
				// corpus_chess_test.go.
				outcome = wr
				if pos.Stm == chess.Black {
					outcome = 1 - wr
				}
				gidTok, haveGid = parts[7], true
				if len(parts) >= 9 {
					if ev, okk := atoiBytes(parts[8]); okk {
						if pos.Stm == chess.Black {
							ev = -ev
						}
						evalCP, hasEval = ev, true
					}
				}
				pos.PackKey(key[:])
			} else {
				if !draughtsAccept(parts, v.Cells) {
					continue
				}
				nLines++
				board := parts[0]

				xm, om, xk, ok := v.Counts(board)
				if v.TBSkip(xm, om, xk, ok) {
					nTBSkip++
					continue
				}

				outcome = 0.5
				if len(parts[2]) == 1 {
					switch parts[2][0] {
					case 'X':
						outcome = 1
					case 'O':
						outcome = 0
					}
				}
				key = packKey(board, v.Cells)
				if len(parts) >= 4 {
					gidTok, haveGid = parts[3], true
				}
				if len(parts) >= 5 {
					if ev, okk := atoiBytes(parts[4]); okk {
						evalCP, hasEval = ev, true
					}
				}
			}

			ai, seen := boardIdx[key]
			if !seen {
				off := int32(len(pool))
				var anchor, nPieces int
				if isChess {
					pool = pos.AppendFeatures(pool)
					anchor = pos.AnchorCP()
					nPieces = pos.PieceCount()
				} else {
					pool, anchor, nPieces = v.AppendFeats(pool, string(parts[0]))
				}

				h := fileHash
				if haveGid {
					h = fnv1aBytes(h, gidTok)
				} else {
					// Legacy 3-column data: fall back to the board itself.
					// After dedup that is still a leak-free split.
					h = fnv1aBytes(fnv1aByte(fnv1aByte(fnv1aInit(), 'b'), ':'), parts[0])
				}

				ai = int32(len(accs))
				accs = append(accs, acc{
					featOff: off, featEnd: int32(len(pool)),
					anchor: int16(anchor), bucket: uint8(v.Bucket(nPieces)),
					isValid: h%uint32(o.Holdout) == 0,
				})
				boardIdx[key] = ai
			}
			a := &accs[ai]
			a.sumOut += outcome
			a.n++
			if hasEval {
				a.sumEval += float64(evalCP)
				a.nEval++
			}
		}
		if err := sc.Err(); err != nil && err != io.EOF {
			fh.Close()
			return nil, fmt.Errorf("read %s: %w", f, err)
		}
		fh.Close()
		if !o.Quiet {
			fmt.Printf("  [%d/%d] %s: %d lines, %d unique so far\n",
				fi+1, len(files), filepath.Base(f), nLines, len(accs))
		}
		if o.MaxLines > 0 && nLines >= int64(o.MaxLines) {
			break
		}
	}

	boardIdx = nil // release the dedup map before training

	sig := func(x float64) float64 { return 1 / (1 + math.Exp(-x)) }
	set := &Set{V: v, Pool: pool}
	var nEvalPos int64
	for i := range accs {
		a := &accs[i]
		target := a.sumOut / float64(a.n)
		if a.nEval > 0 {
			evalAvg := a.sumEval / float64(a.nEval)
			target = o.Lambda*sig(o.KScale*evalAvg) + (1-o.Lambda)*target
			nEvalPos++
		}
		s := Sample{a.featOff, a.featEnd, a.anchor, a.bucket, target}
		if a.isValid {
			set.Valid = append(set.Valid, s)
		} else {
			set.Train = append(set.Train, s)
		}
	}

	set.Stats = Stats{
		Files: len(files), Lines: nLines, Unique: int64(len(accs)),
		Train: int64(len(set.Train)), Valid: int64(len(set.Valid)),
		TBSkip: nTBSkip, EvalPos: nEvalPos,
		Lambda: o.Lambda, KScale: o.KScale,
		Holdout: o.Holdout, GroupKey: o.GroupKey,
	}
	if !o.Quiet {
		set.Stats.Print()
	}
	if len(set.Train) == 0 {
		return nil, fmt.Errorf("no training samples survived loading (check --variant: this data has %d-char boards)", v.Cells)
	}
	return set, nil
}

// Print emits the loader summary in the reference trainer's exact wording.
func (s Stats) Print() {
	dedup := float64(s.Lines) / math.Max(1, float64(s.Unique))
	fmt.Printf("loaded %d lines -> %d UNIQUE positions (%.1fx dedup) -> %d train / %d valid; "+
		"skipped %d TB-territory lines; %d positions had eval labels (lambda=%.2f)\n",
		s.Lines, s.Unique, dedup, s.Train, s.Valid, s.TBSkip, s.EvalPos, s.Lambda)
}

// ── packed corpus (.ntc) ────────────────────────────────────────────────────
//
// Text loading is the slowest part of a Colab run and the only step whose
// result depends on the machine (the holdout key hashes the file PATH, and
// Windows and Linux disagree about separators). Packing solves both: convert
// once on the box where the data lives, upload ~500 MB instead of 3.1 GB, and
// the train/valid split travels frozen inside the file so MSE numbers from two
// machines are directly comparable.

// ntcMagic is the CURRENT packed-corpus format; ntcMagicV1 is still readable.
//
// NTC2 adds two fields the chess kernel needs and the draughts kernel is happy
// to carry: the KERNEL, so a reader knows whether to apply the mirror pair, and
// the K SCALE, because chess's sigmoid is /400 while draughts is /256. Both are
// properties of the corpus, so baking them in keeps train_gpu.py free of
// variant knowledge -- the same reason geometry and the anchor are already in
// there (README.md:70-72).
//
// A new magic rather than more fields under the old one: emit.go:112-117
// records what happened when two different layouts shared the magic "DNN2",
// and the reader has been disambiguating them by file size ever since.
const (
	ntcMagic   = "NTC2"
	ntcMagicV1 = "NTC1"
)

// Save writes the set as a packed corpus.
func (s *Set) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriterSize(f, 1<<20)

	name := make([]byte, 16)
	copy(name, s.V.Name)

	w.WriteString(ntcMagic)
	put32 := func(v uint32) { binary.Write(w, binary.LittleEndian, v) }
	put64 := func(v uint64) { binary.Write(w, binary.LittleEndian, v) }
	putF := func(v float64) { binary.Write(w, binary.LittleEndian, math.Float64bits(v)) }

	put32(2) // version
	w.Write(name)
	put32(uint32(s.V.Kernel))
	put32(uint32(s.V.Cells))
	put32(uint32(s.V.NumFeat))
	put32(uint32(s.V.Buckets))
	put64(uint64(s.Stats.Lines))
	put64(uint64(s.Stats.Unique))
	put64(uint64(len(s.Train)))
	put64(uint64(len(s.Valid)))
	put64(uint64(s.Stats.TBSkip))
	put64(uint64(s.Stats.EvalPos))
	put32(uint32(s.Stats.Holdout))
	putF(s.Stats.Lambda)
	putF(s.Stats.KScale)
	put64(uint64(len(s.Pool)))

	if err := binary.Write(w, binary.LittleEndian, s.Pool); err != nil {
		return err
	}
	for _, part := range [][]Sample{s.Train, s.Valid} {
		for _, sm := range part {
			binary.Write(w, binary.LittleEndian, sm.Off)
			binary.Write(w, binary.LittleEndian, sm.End)
			binary.Write(w, binary.LittleEndian, sm.Anchor)
			w.WriteByte(sm.Bucket)
			binary.Write(w, binary.LittleEndian, math.Float64bits(sm.Y))
		}
	}
	return w.Flush()
}

// NTCHeaderMax is the largest packed-corpus header: 116 bytes for NTC2, 104 for
// NTC1. Reading this many bytes is enough to report every counter a .ntc
// carries, which is what lets cmd/count inventory a 100 MB corpus instantly.
const NTCHeaderMax = 128

// PackedInfo is everything a .ntc header carries.
//
// Note what is NOT in here: a GAME count. The format has never had one, so a
// tool reading only the header cannot report games -- see count.go and the "-"
// in cmd/count's GAMES column for packed sources.
type PackedInfo struct {
	Path      string
	Magic     string
	Version   int
	Name      string
	Kernel    variant.Kernel // meaningful only when Version >= 2
	HasKernel bool
	Cells     int
	NumFeat   int
	Buckets   int
	Stats     Stats
	PoolLen   int64
	HeaderLen int
}

// parseNTCHeader decodes the fixed-size header and returns the bytes consumed.
//
// LoadPacked and PackedHeader share it so the field ORDER lives in exactly one
// place. The variant cross-checks deliberately stay in LoadPacked: it is the
// only one of the two that has a variant to check against.
func parseNTCHeader(path string, raw []byte) (*PackedInfo, int, error) {
	p := 0
	// need guards every read: a truncated header must report "truncated", not
	// panic with an index-out-of-range. A packed corpus is the artifact that
	// travels to Colab and back, so a half-copied file is realistic.
	need := func(n int) error {
		if p+n > len(raw) {
			return fmt.Errorf("%s: header truncated at byte %d (need %d more)", path, p, n)
		}
		return nil
	}
	if err := need(4); err != nil {
		return nil, 0, fmt.Errorf("%s: too short to be a packed corpus", path)
	}
	magic := string(raw[:4])
	if magic != ntcMagic && magic != ntcMagicV1 {
		return nil, 0, fmt.Errorf("%s: not an %s or %s corpus", path, ntcMagic, ntcMagicV1)
	}
	p = 4

	var perr error
	g32 := func() uint32 {
		if perr == nil {
			perr = need(4)
		}
		if perr != nil {
			return 0
		}
		x := binary.LittleEndian.Uint32(raw[p:])
		p += 4
		return x
	}
	g64 := func() uint64 {
		if perr == nil {
			perr = need(8)
		}
		if perr != nil {
			return 0
		}
		x := binary.LittleEndian.Uint64(raw[p:])
		p += 8
		return x
	}
	gF := func() float64 { return math.Float64frombits(g64()) }

	ver := g32()
	if perr != nil {
		return nil, 0, perr
	}
	if (magic == ntcMagicV1 && ver != 1) || (magic == ntcMagic && ver != 2) {
		return nil, 0, fmt.Errorf("%s: unsupported %s version %d", path, magic, ver)
	}
	if err := need(16); err != nil {
		return nil, 0, err
	}
	nameRaw := raw[p : p+16]
	p += 16
	info := &PackedInfo{
		Path:    path,
		Magic:   magic,
		Version: int(ver),
		Name:    string(nameRaw[:clen(nameRaw)]),
	}
	if ver >= 2 {
		info.Kernel, info.HasKernel = variant.Kernel(g32()), true
	}
	info.Cells, info.NumFeat, info.Buckets = int(g32()), int(g32()), int(g32())

	st := Stats{Lambda: 0, Holdout: 0}
	st.Lines = int64(g64())
	st.Unique = int64(g64())
	st.Train = int64(g64())
	st.Valid = int64(g64())
	st.TBSkip = int64(g64())
	st.EvalPos = int64(g64())
	st.Holdout = int(g32())
	st.Lambda = gF()
	if ver >= 2 {
		st.KScale = gF()
	}
	info.Stats = st
	info.PoolLen = int64(g64())
	if perr != nil {
		return nil, 0, perr
	}
	info.HeaderLen = p
	return info, p, nil
}

// PackedHeader reports a packed corpus's counters without reading its body --
// data\chess.ntc is ~100 MB and the header is 116 bytes.
//
// It does NOT check the file against a variant (that is LoadPacked's job) and it
// does NOT validate the body length, because it never reads the body. A caller
// that needs the data itself must use LoadPacked.
func PackedHeader(path string) (*PackedInfo, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	buf := make([]byte, NTCHeaderMax)
	// A valid NTC1 file with an empty body is shorter than NTCHeaderMax, so a
	// short read is not an error here -- parseNTCHeader's own bounds checks
	// decide whether what arrived is enough.
	n, err := io.ReadFull(f, buf)
	if err != nil && err != io.ErrUnexpectedEOF && err != io.EOF {
		return nil, err
	}
	info, _, err := parseNTCHeader(path, buf[:n])
	return info, err
}

// LoadPacked reads a packed corpus and verifies it matches the variant.
func LoadPacked(v *variant.Variant, path string, quiet bool) (*Set, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	info, p, err := parseNTCHeader(path, raw)
	if err != nil {
		return nil, err
	}
	name := info.Name
	if name != v.Name {
		return nil, fmt.Errorf("%s holds variant %q but --variant is %q", path, name, v.Name)
	}
	if info.HasKernel {
		if k := info.Kernel; k != v.Kernel {
			return nil, fmt.Errorf("%s was packed for the %s kernel but %s uses %s",
				path, k, v.Name, v.Kernel)
		}
	} else if v.IsChess() {
		// NTC1 predates the chess kernel entirely, so a v1 file claiming to be
		// chess is corrupt or hand-edited.
		return nil, fmt.Errorf("%s is an %s corpus, which cannot hold chess data; repack it",
			path, ntcMagicV1)
	}
	cells, numFeat, buckets := info.Cells, info.NumFeat, info.Buckets
	if cells != v.Cells || numFeat != v.NumFeat {
		return nil, fmt.Errorf("%s: geometry mismatch (cells %d/%d, feats %d/%d)",
			path, cells, v.Cells, numFeat, v.NumFeat)
	}
	st := info.Stats
	nTrain, nValid := st.Train, st.Valid
	poolLen := int(info.PoolLen)

	// Check the body length BEFORE reading it. The trailing-byte check at the
	// end only catches a file that is too long; a SHORT one would run off the
	// end of raw and panic with an index-out-of-range instead of saying the
	// corpus is truncated. A packed corpus is the artifact that travels to
	// Colab and back, so a half-copied file is a realistic thing to be handed.
	const recBytes = 4 + 4 + 2 + 1 + 8 // off, end, anchor, bucket, y
	if poolLen < 0 || nTrain < 0 || nValid < 0 {
		return nil, fmt.Errorf("%s: negative length in header (corrupt)", path)
	}
	want := int64(p) + int64(poolLen)*2 + (nTrain+nValid)*recBytes
	if int64(len(raw)) != want {
		return nil, fmt.Errorf("%s: body is %d bytes, header describes %d "+
			"(truncated or corrupt)", path, int64(len(raw))-int64(p), want-int64(p))
	}

	set := &Set{V: v, Stats: st}
	set.Pool = make([]uint16, poolLen)
	for i := 0; i < poolLen; i++ {
		set.Pool[i] = binary.LittleEndian.Uint16(raw[p:])
		p += 2
	}
	readSamples := func(n int64) []Sample {
		out := make([]Sample, n)
		for i := range out {
			out[i].Off = int32(binary.LittleEndian.Uint32(raw[p:]))
			p += 4
			out[i].End = int32(binary.LittleEndian.Uint32(raw[p:]))
			p += 4
			out[i].Anchor = int16(binary.LittleEndian.Uint16(raw[p:]))
			p += 2
			out[i].Bucket = raw[p]
			p++
			out[i].Y = math.Float64frombits(binary.LittleEndian.Uint64(raw[p:]))
			p += 8
		}
		return out
	}
	set.Train = readSamples(nTrain)
	set.Valid = readSamples(nValid)
	if p != len(raw) {
		return nil, fmt.Errorf("%s: trailing %d bytes (corrupt or truncated)", path, len(raw)-p)
	}
	if !quiet {
		fmt.Printf("packed corpus %s: variant=%s cells=%d feats=%d buckets=%d\n",
			filepath.Base(path), name, cells, numFeat, buckets)
		set.Stats.Print()
	}
	return set, nil
}

func clen(b []byte) int {
	for i, c := range b {
		if c == 0 {
			return i
		}
	}
	return len(b)
}
