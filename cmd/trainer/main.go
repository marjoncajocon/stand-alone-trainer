// Command trainer trains an NNUE for any of the six checkers/draughts engines
// in this workspace and emits a drop-in nnue_weights.h.
//
//	trainer --variant=filipino --data="...\data\selfplay_*.txt" --h=256
//	trainer --variant=filipino --corpus=filipino.ntc --gpu=1 --algo=minibatch
//	trainer --variant=filipino --load=nn.ntw --probe=<board>
//
// See README.md for the validation gate chain. The short version: never trust a
// net until probe parity passes, probe_accbound clears, and the holdout MSE
// beats the incumbent.
package main

import (
	"flag"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"time"

	"nnuetrainer/internal/corpus"
	"nnuetrainer/internal/emit"
	"nnuetrainer/internal/gpu"
	"nnuetrainer/internal/model"
	"nnuetrainer/internal/train"
	"nnuetrainer/internal/variant"
)

type multiFlag []string

func (m *multiFlag) String() string     { return strings.Join(*m, ",") }
func (m *multiFlag) Set(s string) error { *m = append(*m, s); return nil }

func main() {
	var data multiFlag

	variantName := flag.String("variant", "filipino", "which engine to train for; --list-variants to see all")
	listVariants := flag.Bool("list-variants", false, "print the variant table and exit")
	flag.Var(&data, "data", "glob of selfplay text files (repeatable). "+
		"Default: data\\<variant>\\*.txt beside the binary, or a packed data\\<variant>.ntc")
	corpusPath := flag.String("corpus", "", "packed .ntc corpus (mutually exclusive with --data)")
	pack := flag.Bool("pack", false, "convert --data into --out-corpus and exit")
	outCorpus := flag.String("out-corpus", "corpus.ntc", "output path for --pack")

	h := flag.Int("h", 0, "hidden units (0 = the variant's shipped NN_H)")
	buckets := flag.Int("buckets", 0, "W2 phase buckets (0 = what the engine's kernel implements)")

	algo := flag.String("algo", "online", "online (batch-1 AdaGrad reference) or minibatch (production, GPU-capable)")
	opt := flag.String("opt", "", "adam or adagrad (default: adagrad for online, adam for minibatch)")
	epochs := flag.Int("epochs", 0, "training epochs (default 14 online, 40 minibatch)")
	lr := flag.Float64("lr", 0, "learning rate (default 0.05 online, 0.001 minibatch)")
	batch := flag.Int("batch", 8192, "minibatch size")
	schedule := flag.String("lr-schedule", "", "const or cosine (default const online, cosine minibatch)")
	warmup := flag.Int("warmup-epochs", -1, "LR warmup epochs (default 0 online, 1 minibatch)")
	kScale := flag.Float64("k", 1.0/256.0, "sigmoid scale K")
	lambda := flag.Float64("lambda", 0.5, "eval-label blend: y = lambda*sigmoid(K*eval) + (1-lambda)*outcome")
	holdout := flag.Int("holdout", 20, "1-in-N GAMES held out for validation")
	seed := flag.Int64("seed", 42, "RNG seed for weight init and epoch shuffling")
	maxLines := flag.Int("max-lines", 0, "cap on data lines (0 = all; smoke tests)")
	groupKey := flag.String("group-key", "rel", "holdout group key: rel (<variant>/<file>, portable), "+
		"path (what the reference trainer hashes; NOT portable across machines), or basename")

	gpuFlag := flag.String("gpu", "0", "0 = CPU, 1 = REQUIRE CUDA (exit 3 if absent), auto = use if present")
	device := flag.Int("device", 0, "CUDA device ordinal")
	threads := flag.Int("threads", 0, "minibatch worker goroutines (0 = NumCPU); does NOT change results")

	out := flag.String("out", "", "quantized weights output (default nn_<variant>_h<H>.ntw)")
	outH := flag.String("outh", "nnue_weights.h", "generated C header")
	logPath := flag.String("log", "", "tee all output to this file (DO THIS: the holdout MSE has been lost twice)")
	repo := flag.String("repo", "", "workspace root holding the engine repos (default: auto-detect)")

	load := flag.String("load", "", "load a weights file instead of training")
	probe := flag.String("probe", "", "with --load: print score16 for this board and exit")
	probeNarrow := flag.String("probe-int16", "", "same, but through the engine's narrow int16 accumulator")
	probeFile := flag.String("probe-file", "", "with --load: score every board in this file, one score16 per line")
	probeNarrowFile := flag.Bool("probe-file-int16", false, "make --probe-file use the narrow int16 accumulator")
	convert := flag.Bool("convert", false, "with --load: re-emit --outh/--out from an existing weights file and exit "+
		"(reads legacy DNN1/DNN2 too, so a historical net can be turned into a header)")
	evalOnly := flag.Bool("eval-only", false, "with --load: report holdout MSE of an existing net")

	flag.Parse()

	if *listVariants {
		printVariants()
		return
	}

	// An UNQUOTED glob on a Unix shell is expanded by the shell, so only the
	// first match reaches --data and every other filename lands here as a
	// positional argument. Go's flag package stops parsing at the first
	// non-flag, so the run would quietly train on ONE file out of 168 and
	// report it in a line that is easy to skim past. Refuse instead.
	if flag.NArg() > 0 {
		die(2, "unexpected argument %q (and %d more).\n"+
			"  If you passed a glob, QUOTE it so this program expands it, not the shell:\n"+
			"    --data \"data/%s/*.txt\"     (correct)\n"+
			"    --data data/%s/*.txt       (shell expands it; only 1 file is read)",
			flag.Arg(0), flag.NArg()-1, *variantName, *variantName)
	}

	if *logPath != "" {
		stop := teeStdout(*logPath)
		defer stop()
	}

	v, err := variant.Get(*variantName)
	if err != nil {
		die(2, "%v", err)
	}
	if *buckets == 0 {
		*buckets = v.Buckets
	}
	if *buckets > v.Buckets {
		warn("--buckets=%d but %s's search.c implements %d. The generated header will\n"+
			"        NOT COMPILE against that engine (int16_t(*)[H] vs int16_t*). Add an\n"+
			"        #ifdef NN_W2_BUCKETS branch to %s/search.c and its probe_nnue.c first.",
			*buckets, v.Name, v.Buckets, v.EngineDir)
	}

	workspace := resolveRepo(*repo, v)

	// ── probe modes: no data needed ─────────────────────────────────────────
	if *load != "" && (*probe != "" || *probeNarrow != "") {
		board := *probe
		narrow := false
		if *probeNarrow != "" {
			board, narrow = *probeNarrow, true
		}
		q, err := emit.LoadBin(*load, v)
		if err != nil {
			die(2, "load: %v", err)
		}
		if len(board) != v.Cells {
			die(2, "board must be %d chars for variant %s (got %d)", v.Cells, v.Name, len(board))
		}
		feats, _, pieces := v.AppendFeats(nil, board)
		var sc int
		if narrow {
			sc = q.Score16Narrow(feats, v.Bucket(pieces))
		} else {
			sc = q.Score16(feats, v.Bucket(pieces))
		}
		fmt.Printf("score16=%d\n", sc)
		return
	}

	// Batch probe: one score per line, so a 350-board parity sweep costs one
	// process instead of 350.
	if *load != "" && *probeFile != "" {
		q, err := emit.LoadBin(*load, v)
		if err != nil {
			die(2, "load: %v", err)
		}
		raw, err := os.ReadFile(*probeFile)
		if err != nil {
			die(2, "probe-file: %v", err)
		}
		// stdout carries ONLY the scores, one per line, and nothing goes to
		// stderr: PowerShell 5.1 turns any native-command stderr write into a
		// terminating NativeCommandError under $ErrorActionPreference='Stop',
		// which would break tools/parity.ps1 for a progress message nobody needs.
		for _, line := range strings.Split(strings.ReplaceAll(string(raw), "\r\n", "\n"), "\n") {
			line = strings.TrimSpace(line)
			if len(line) != v.Cells {
				continue
			}
			feats, _, pieces := v.AppendFeats(nil, line)
			if *probeNarrowFile {
				fmt.Printf("%d\n", q.Score16Narrow(feats, v.Bucket(pieces)))
			} else {
				fmt.Printf("%d\n", q.Score16(feats, v.Bucket(pieces)))
			}
		}
		return
	}

	// Convert an existing weights file into a header without retraining. This
	// is how a historical net gets a header emitted by THIS trainer, which is
	// what makes a C-side parity check meaningful.
	if *load != "" && *convert {
		q, err := emit.LoadBin(*load, v)
		if err != nil {
			die(2, "load: %v", err)
		}
		if err := emit.Header(*outH, q, v); err != nil {
			die(2, "write header: %v", err)
		}
		if *out != "" {
			if err := emit.Bin(*out, q, v); err != nil {
				die(2, "write weights: %v", err)
			}
		}
		fmt.Printf("converted %s -> %s (variant=%s H=%d feats=%d buckets=%d)\n",
			*load, *outH, v.Name, q.H, q.NumFeat, q.Buckets)
		return
	}

	// ── resolve the compute backend BEFORE touching any data ────────────────
	//
	// This check must come first. Loading the corpus takes minutes on a 3 GB
	// text set, and failing after that for a flag we could have rejected
	// immediately is just a slow way to say no.
	minib := *algo == "minibatch"
	useGPU := false
	switch *gpuFlag {
	case "0", "false", "":
	case "1", "true":
		if !gpu.Available() {
			fmt.Fprintln(os.Stderr, "error: "+gpu.ErrNoCUDA.Error())
			os.Exit(3)
		}
		useGPU = true
	case "auto":
		useGPU = gpu.Available()
		if !useGPU {
			fmt.Println("note: --gpu=auto and no CUDA device; running on CPU.")
		}
	default:
		die(2, "--gpu must be 0, 1 or auto (got %q)", *gpuFlag)
	}
	if useGPU && !minib {
		die(2, "--gpu requires --algo=minibatch.\n"+
			"  The online path is batch-size-1 by definition: one sequential update per\n"+
			"  position, each touching ~14 feature rows. That is latency-bound and would\n"+
			"  run SLOWER on a GPU than on a CPU. Minibatching is what makes a GPU useful.")
	}

	// ── corpus ──────────────────────────────────────────────────────────────
	if *corpusPath != "" && len(data) > 0 {
		die(2, "--corpus and --data are mutually exclusive")
	}
	// With neither flag given, look in the standard place: data/<variant>/ next
	// to the binary, exactly like stand-alone-selfplay's --out-dir default. A
	// packed data/<variant>.ntc wins if present, since it is both faster and
	// split-frozen.
	if *corpusPath == "" && len(data) == 0 {
		if pk, glob, where := discoverData(v); pk != "" {
			*corpusPath = pk
			fmt.Printf("using packed corpus %s\n", pk)
		} else if glob != "" {
			data = append(data, glob)
			fmt.Printf("using %s\n", glob)
		} else {
			die(2, "no data found.\n"+
				"  Put selfplay files in one of these, or pass --data / --corpus:\n%s"+
				"  Expected  data\\%s\\*.txt  or a packed  data\\%s.ntc\n"+
				"  (--list-variants shows every variant)", where, v.Name, v.Name)
		}
	}

	var set *corpus.Set
	switch {
	case *corpusPath != "":
		set, err = corpus.LoadPacked(v, *corpusPath, false)
	default:
		set, err = corpus.Load(v, corpus.Options{
			Globs: data, Holdout: *holdout, Lambda: *lambda, KScale: *kScale,
			MaxLines: *maxLines, GroupKey: *groupKey,
		})
	}
	if err != nil {
		die(2, "%v", err)
	}

	if *pack {
		if err := set.Save(*outCorpus); err != nil {
			die(2, "pack: %v", err)
		}
		fi, _ := os.Stat(*outCorpus)
		fmt.Printf("wrote %s (%.1f MB). The train/valid split is FROZEN inside it, so a run\n"+
			"on another machine is directly comparable to this one.\n",
			*outCorpus, float64(fi.Size())/(1<<20))
		return
	}

	fmt.Printf("valid MSE, anchor-only (material): %.6f\n", model.AnchorMSE(set.Valid, *kScale))

	if *load != "" && *evalOnly {
		q, err := emit.LoadBin(*load, v)
		if err != nil {
			die(2, "load: %v", err)
		}
		fmt.Printf("valid MSE, QUANTIZED net (%s): %.6f\n", *load, q.MSE(set.Pool, set.Valid, *kScale))
		return
	}

	// ── defaults that depend on the algorithm ───────────────────────────────
	if *epochs == 0 {
		*epochs = pick(minib, 40, 14)
	}
	if *lr == 0 {
		*lr = pickF(minib, 0.001, 0.05)
	}
	if *opt == "" {
		*opt = pickS(minib, "adam", "adagrad")
	}
	if *schedule == "" {
		*schedule = pickS(minib, "cosine", "const")
	}
	if *warmup < 0 {
		*warmup = pick(minib, 1, 0)
	}

	// H is resolved HERE, not earlier: --pack, --probe, --probe-file, --convert
	// and --eval-only all take their architecture from the file they are given,
	// so requiring H before those would fail a pack on any machine that has no
	// engine repo to read NN_H from (e.g. Colab).
	if *h == 0 {
		*h = shippedH(workspace, v)
	}
	if *h <= 0 {
		die(2, "could not determine H (no %s/nnue_weights.h found to read NN_H from).\n"+
			"  Pass --h explicitly, e.g. --h 256.", v.EngineDir)
	}
	if *h%8 != 0 {
		warn("H=%d is not a multiple of 8; the shipped kernel's inner loop vectorizes poorly.", *h)
	}
	if sh := shippedH(workspace, v); sh > 0 && sh != *h {
		warn("training H=%d but %s/nnue_weights.h ships NN_H %d. The engine reads NN_H from\n"+
			"        whatever header it is given, so installing this net CHANGES the architecture.",
			*h, v.EngineDir, sh)
	}

	rng := rand.New(rand.NewSource(*seed))
	m := model.New(v, *h, *buckets, rng)

	fmt.Printf("variant=%s  H=%d  feats=%d  buckets=%d  algo=%s  opt=%s  epochs=%d  lr=%g  seed=%d\n",
		v.Name, *h, v.NumFeat, *buckets, *algo, *opt, *epochs, *lr, *seed)
	fmt.Printf("backend=%s  threads=%d  TB-skip: %s\n",
		gpu.Backend(), effThreads(*threads), v.SkipDoc)

	t0 := time.Now()
	if minib {
		runMinibatch(m, set, v, useGPU, *device, train.BatchOpts{
			Epochs: *epochs, Batch: *batch, LR: float32(*lr), K: float32(*kScale),
			Opt: *opt, Schedule: *schedule, WarmupEpochs: *warmup,
			Threads: *threads, Rng: rng,
		})
	} else {
		train.Online(m, set, train.OnlineOpts{
			Epochs: *epochs, LR: *lr, K: *kScale, Rng: rng,
			Report: func(ep int, mse float64, el time.Duration) {
				fmt.Printf("epoch %d: valid MSE %.6f  (%.1fs)\n", ep, mse, el.Seconds())
			},
		})
	}

	// ── quantize, report, emit ──────────────────────────────────────────────
	q := m.Quantize()
	fmt.Printf("valid MSE, QUANTIZED net: %.6f\n", q.MSE(set.Pool, set.Valid, *kScale))

	lo, hi := q.AccBound(set.Pool, set.Valid)
	head := int32(32767)
	fmt.Printf("accumulator bound over the holdout: HI %+d  LO %+d  (int16 limit +/-32767)\n", hi, lo)
	if hi > head || lo < -head {
		die(4, "ACCUMULATOR OVERFLOW: this net would silently miscompute in the engine's\n"+
			"  int16 kernel. Do not install it. Retrain with a smaller --lr or add clipping.")
	}
	fmt.Printf("  headroom: %d above HI, %d below LO — int16 accumulator is SAFE.\n", head-hi, head+lo)
	fmt.Println("  (still run c_port/tools/probes/probe_accbound.exe: it scans a different set)")

	if *out == "" {
		*out = fmt.Sprintf("nn_%s_h%d.ntw", v.Name, *h)
	}
	if err := emit.Bin(*out, q, v); err != nil {
		die(2, "write weights: %v", err)
	}
	if err := emit.Header(*outH, q, v); err != nil {
		die(2, "write header: %v", err)
	}
	fmt.Printf("wrote %s + %s (variant=%s H=%d buckets=%d) in %s\n",
		*out, *outH, v.Name, *h, *buckets, time.Since(t0).Round(time.Second))
	fmt.Printf("install with: copy %s %s\\nnue_weights.h\n", *outH,
		filepath.FromSlash(filepath.Join(workspace, v.EngineDir)))
}

func runMinibatch(m *model.Model, set *corpus.Set, v *variant.Variant,
	useGPU bool, ordinal int, o train.BatchOpts) {

	b := train.NewBatch(m)
	if !useGPU {
		b.Run(set, o)
		b.Into(m)
		return
	}

	dev, err := gpu.Open(ordinal)
	if err != nil {
		die(3, "gpu: %v", err)
	}
	defer dev.Close()
	fmt.Printf("gpu: %s\n", dev.Name())

	gc := packForGPU(set)
	w := &gpu.Weights{W1: b.W1, B1: b.B1, W2: b.W2}
	p := &gpu.Params{H: m.H, NumFeat: m.NumFeat, Buckets: m.Buckets,
		Mirror: m.Mirror, K: o.K, Batch: o.Batch, Opt: o.Opt}
	if err := dev.Upload(gc, w, p); err != nil {
		die(3, "gpu upload: %v", err)
	}

	perm := make([]int32, len(set.Train))
	for i := range perm {
		perm[i] = int32(i)
	}
	for ep := 1; ep <= o.Epochs; ep++ {
		t0 := time.Now()
		o.Rng.Shuffle(len(perm), func(i, j int) { perm[i], perm[j] = perm[j], perm[i] })
		lrEp := train.ScheduleLR(o, ep)
		if err := dev.RunEpoch(perm, lrEp); err != nil {
			die(3, "gpu epoch %d: %v", ep, err)
		}
		mse, err := dev.ValidMSE(gc)
		if err != nil {
			die(3, "gpu valid: %v", err)
		}
		fmt.Printf("epoch %d: valid MSE %.6f  (lr %.2e, %.1fs)\n", ep, mse, lrEp, time.Since(t0).Seconds())
	}
	if err := dev.Download(w); err != nil {
		die(3, "gpu download: %v", err)
	}
	b.Into(m)
}

// packForGPU flattens the set into the device layout: TRAIN FIRST, then VALID,
// so the holdout is one contiguous range.
func packForGPU(set *corpus.Set) *gpu.Corpus {
	n := len(set.Train) + len(set.Valid)
	c := &gpu.Corpus{
		Pool:   set.Pool,
		Off:    make([]int32, n),
		End:    make([]int32, n),
		Anchor: make([]int16, n),
		Bucket: make([]uint8, n),
		Y:      make([]float32, n),

		TrainCount: len(set.Train),
		ValidCount: len(set.Valid),
	}
	i := 0
	for _, part := range [][]corpus.Sample{set.Train, set.Valid} {
		for _, s := range part {
			c.Off[i], c.End[i] = s.Off, s.End
			c.Anchor[i] = s.Anchor
			c.Bucket[i] = s.Bucket
			c.Y[i] = float32(s.Y)
			i++
		}
	}
	return c
}

// ── helpers ─────────────────────────────────────────────────────────────────

func printVariants() {
	fmt.Printf("%-14s %-30s %5s %6s %7s %8s %8s  %s\n",
		"VARIANT", "ENGINE", "CELLS", "FEATS", "BUCKETS", "MAN", "KING", "TB-SKIP")
	for _, n := range variant.Names() {
		v, _ := variant.Get(n)
		fmt.Printf("%-14s %-30s %5d %6d %7d %8d %8d  %s\n",
			v.Name, v.EngineDir, v.Cells, v.NumFeat, v.Buckets, v.ValMan, v.ValKing, v.SkipDoc)
	}
	fmt.Println("\nBUCKETS is what that engine's search.c implements today, not a preference.")
	fmt.Println("english/russian/international have no NN_W2_BUCKETS branch at all.")
}

var reNNH = regexp.MustCompile(`(?m)^#define\s+NN_H\s+(\d+)`)

// shippedH reads NN_H out of the engine's installed header, so a bare run
// trains the architecture that engine actually has.
func shippedH(workspace string, v *variant.Variant) int {
	if workspace == "" {
		return 0
	}
	raw, err := os.ReadFile(filepath.Join(workspace, filepath.FromSlash(v.EngineDir), "nnue_weights.h"))
	if err != nil {
		return 0
	}
	mm := reNNH.FindSubmatch(raw)
	if mm == nil {
		return 0
	}
	n := 0
	for _, c := range mm[1] {
		n = n*10 + int(c-'0')
	}
	return n
}

// dataRoots lists where a corpus is looked for, in priority order: beside the
// binary first (bin\data\<variant>, the same convention stand-alone-selfplay
// uses for --out-dir), then the kit root, then the current directory.
func dataRoots() []string {
	var roots []string
	if exe, err := os.Executable(); err == nil {
		d := filepath.Dir(exe)
		roots = append(roots, filepath.Join(d, "data"), filepath.Join(d, "..", "data"))
	}
	if wd, err := os.Getwd(); err == nil {
		roots = append(roots, filepath.Join(wd, "data"))
	}
	seen := map[string]bool{}
	var out []string
	for _, r := range roots {
		if abs, err := filepath.Abs(r); err == nil && !seen[abs] {
			seen[abs] = true
			out = append(out, abs)
		}
	}
	return out
}

// discoverData finds a corpus for v without any flags. Returns (packedPath,
// textGlob, searchedDescription); at most one of the first two is non-empty.
func discoverData(v *variant.Variant) (string, string, string) {
	var searched strings.Builder
	for _, root := range dataRoots() {
		fmt.Fprintf(&searched, "    %s\n", root)
		// A packed corpus wins: faster to load and its split is frozen.
		pk := filepath.Join(root, v.Name+".ntc")
		if _, err := os.Stat(pk); err == nil {
			return pk, "", ""
		}
		glob := filepath.Join(root, v.Name, "*.txt")
		if m, err := filepath.Glob(glob); err == nil && len(m) > 0 {
			return "", glob, ""
		}
	}
	return "", "", searched.String()
}

// resolveRepo finds the workspace root that holds the engine repos.
func resolveRepo(explicit string, v *variant.Variant) string {
	// An explicit --repo is AUTHORITATIVE. Falling through to the guesses when
	// it does not contain the engine would make the flag silently do nothing.
	if explicit != "" {
		if _, err := os.Stat(filepath.Join(explicit, filepath.FromSlash(v.EngineDir))); err == nil {
			abs, _ := filepath.Abs(explicit)
			return abs
		}
		return ""
	}
	cands := []string{}
	if wd, err := os.Getwd(); err == nil {
		cands = append(cands, filepath.Join(wd, ".."), wd, filepath.Join(wd, "..", ".."))
	}
	if exe, err := os.Executable(); err == nil {
		d := filepath.Dir(exe)
		cands = append(cands, filepath.Join(d, ".."), filepath.Join(d, "..", ".."))
	}
	for _, c := range cands {
		if _, err := os.Stat(filepath.Join(c, filepath.FromSlash(v.EngineDir))); err == nil {
			abs, _ := filepath.Abs(c)
			return abs
		}
	}
	return ""
}

// teeStdout duplicates everything written to stdout into path. This exists
// because the decisive holdout MSE has now been lost twice by being printed to
// a console that scrolled away.
func teeStdout(path string) func() {
	f, err := os.Create(path)
	if err != nil {
		die(2, "log: %v", err)
	}
	r, w, err := os.Pipe()
	if err != nil {
		die(2, "log pipe: %v", err)
	}
	orig := os.Stdout
	os.Stdout = w
	done := make(chan struct{})
	go func() {
		io.Copy(io.MultiWriter(orig, f), r)
		close(done)
	}()
	return func() {
		w.Close()
		<-done
		f.Close()
		os.Stdout = orig
	}
}

func effThreads(t int) int {
	if t <= 0 {
		return runtime.NumCPU()
	}
	return t
}

func pick(cond bool, a, b int) int {
	if cond {
		return a
	}
	return b
}

func pickS(cond bool, a, b string) string {
	if cond {
		return a
	}
	return b
}

func pickF(cond bool, a, b float64) float64 {
	if cond {
		return a
	}
	return b
}

func warn(f string, a ...any) { fmt.Fprintf(os.Stderr, "WARNING: "+f+"\n", a...) }

func die(code int, f string, a ...any) {
	fmt.Fprintf(os.Stderr, "error: "+f+"\n", a...)
	os.Exit(code)
}
