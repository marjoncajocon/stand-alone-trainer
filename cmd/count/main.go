// Command count reports how much selfplay data this kit has, per variant.
//
//	count                                     every variant that has data
//	count --variant=filipino --files          per-file breakdown too
//	count --variant=chess                     reads data\chess.ntc's header only
//	count --variant=filipino --data "D:\...\data\filipino\*.txt" --json
//
// It replaces the engine kits' count.ps1. Two differences worth knowing:
// GAMES comes from the data (distinct gameids per file) rather than from
// grepping "done:" in the paired .log, so it still works when the .log is gone;
// and the accept/reject rules are literally the trainer's own, so LINES,
// UNIQUE and TB-SKIP are what training would load, not an approximation.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"nnuetrainer/internal/corpus"
	"nnuetrainer/internal/datadir"
	"nnuetrainer/internal/variant"
)

// buildHash and buildDate are injected by build_all.ps1's -ldflags -X. They must
// be declared here for that to land anywhere.
var buildHash, buildDate string

type multiFlag []string

func (m *multiFlag) String() string     { return strings.Join(*m, ",") }
func (m *multiFlag) Set(s string) error { *m = append(*m, s); return nil }

// row is one variant's line in the table.
//
// Games is -1 when the source cannot answer it (a packed .ntc header carries no
// game count), which is a different thing from 0.
type row struct {
	Variant   string    `json:"variant"`
	Source    string    `json:"source"`
	FileCount int       `json:"files"`
	Games     int64     `json:"games"`
	Lines     int64     `json:"lines"`
	Unique    int64     `json:"unique"`
	Malformed int64     `json:"malformed"`
	TBSkip    int64     `json:"tbSkip"`
	NoGameID  int64     `json:"noGameId"`
	Exact     bool      `json:"exactUnique"`
	PerFile   []fileRow `json:"perFile,omitempty"`
}

type fileRow struct {
	Path      string `json:"path"`
	Games     int64  `json:"games"`
	Lines     int64  `json:"lines"`
	Unique    int64  `json:"unique"`
	Malformed int64  `json:"malformed"`
	TBSkip    int64  `json:"tbSkip"`
	Error     string `json:"error,omitempty"`
}

type report struct {
	BuildHash string   `json:"buildHash,omitempty"`
	BuildDate string   `json:"buildDate,omitempty"`
	Variants  []row    `json:"variants"`
	Total     row      `json:"total"`
	NoData    []string `json:"noData"`
	Errors    []string `json:"errors"`
}

const (
	srcText   = "text"
	srcPacked = "packed(.ntc)"
)

func main() {
	var data multiFlag

	variantNames := flag.String("variant", "", "comma-separated variants to count (default: every variant that has data)")
	flag.Var(&data, "data", "glob of selfplay text files (repeatable); needs exactly one --variant")
	corpusPath := flag.String("corpus", "", "packed .ntc to read the header of; needs exactly one --variant")
	perFile := flag.Bool("files", false, "also list every input file with its own counts")
	asJSON := flag.Bool("json", false, "emit one JSON object and nothing else")
	threads := flag.Int("threads", 0, "file-scanning workers (0 = NumCPU, capped at the file count)")
	unique := flag.Bool("unique", true, "count UNIQUE positions; false is faster and uses no memory for the set")
	exact := flag.Bool("exact-unique", false, "dedup on exact pack keys instead of a 64-bit hash (~5x the memory)")
	version := flag.Bool("version", false, "print the build stamp and exit")
	// Deliberately no --max-lines. The trainer's version stops mid-file, which
	// under per-file parallel workers would make the per-file table depend on
	// goroutine scheduling.
	flag.Parse()

	if *version {
		h, d := buildHash, buildDate
		if h == "" {
			h = "unstamped (built by hand, not build_all.ps1)"
		}
		if d == "" {
			d = "unknown"
		}
		fmt.Printf("count  source hash %s  built %s  %s/%s\n", h, d, runtime.GOOS, runtime.GOARCH)
		return
	}

	// An UNQUOTED glob on a Unix shell is expanded by the shell, so only the
	// first match reaches --data and every other filename lands here as a
	// positional argument -- and the count would silently be of ONE file.
	if flag.NArg() > 0 {
		die(2, "unexpected argument %q (and %d more).\n"+
			"  If you passed a glob, QUOTE it so this program expands it, not the shell:\n"+
			"    --data \"data/filipino/*.txt\"     (correct)\n"+
			"    --data data/filipino/*.txt       (shell expands it; only 1 file is read)",
			flag.Arg(0), flag.NArg()-1)
	}
	if *corpusPath != "" && len(data) > 0 {
		die(2, "--corpus and --data are mutually exclusive")
	}

	var want []*variant.Variant
	if *variantNames == "" {
		for _, n := range variant.Names() {
			v, err := variant.Get(n)
			if err != nil {
				die(2, "%v", err)
			}
			want = append(want, v)
		}
	} else {
		for _, n := range strings.Split(*variantNames, ",") {
			n = strings.TrimSpace(n)
			if n == "" {
				continue
			}
			v, err := variant.Get(n)
			if err != nil {
				die(2, "%v", err)
			}
			want = append(want, v)
		}
	}
	if len(want) == 0 {
		die(2, "--variant listed no variants")
	}
	// An explicit source describes ONE corpus, so it cannot be spread over a
	// list of variants: the geometry check would reject all but one anyway.
	if (len(data) > 0 || *corpusPath != "") && len(want) != 1 {
		die(2, "--data/--corpus needs exactly one --variant (got %d)", len(want))
	}

	explicit := *variantNames != ""
	opts := corpus.CountOptions{
		Threads: *threads,
		Unique:  *unique,
		Exact:   *exact,
		// Always retained: it costs one struct per file and is the only way to
		// name a file that failed to read.
		PerFile: true,
	}

	rep := report{BuildHash: buildHash, BuildDate: buildDate}
	for _, v := range want {
		r, err := countOne(v, data, *corpusPath, opts)
		if err != nil {
			// With an explicit --variant this is fatal: the user named a source
			// and it is not there. In a sweep it just means "nothing here".
			if explicit {
				die(2, "%v", err)
			}
			rep.NoData = append(rep.NoData, v.Name)
			continue
		}
		for _, fr := range r.PerFile {
			if fr.Error != "" {
				rep.Errors = append(rep.Errors, fmt.Sprintf("%s: %s", fr.Path, fr.Error))
			}
		}
		rep.Variants = append(rep.Variants, *r)
	}

	if len(rep.Variants) == 0 {
		if *asJSON {
			rep.Total = total(rep.Variants, *exact)
			emitJSON(rep)
		} else {
			fmt.Println("no data found in any variant.")
			fmt.Print(searchedNote())
		}
		os.Exit(5)
	}
	rep.Total = total(rep.Variants, *exact)

	if *asJSON {
		if !*perFile {
			for i := range rep.Variants {
				rep.Variants[i].PerFile = nil
			}
		}
		emitJSON(rep)
	} else {
		printTable(rep, *perFile, *unique, *exact)
	}
	if len(rep.Errors) > 0 {
		os.Exit(1)
	}
}

// countOne resolves a variant's source and counts it.
func countOne(v *variant.Variant, data []string, corpusPath string,
	o corpus.CountOptions) (*row, error) {

	globs := data
	packed := corpusPath
	if len(globs) == 0 && packed == "" {
		pk, g, searched := datadir.Discover(v)
		switch {
		case pk != "":
			packed = pk
		case len(g) > 0:
			globs = g
		default:
			var want strings.Builder
			for _, f := range datadir.Folders(v) {
				fmt.Fprintf(&want, "  Expected  data\\%s\\*.txt\n", f)
			}
			return nil, fmt.Errorf("no %s data found.\n"+
				"  Put selfplay files in one of these, or pass --data / --corpus:\n%s"+
				"%s  or a packed  data\\%s.ntc", v.Name, searched, want.String(), v.Name)
		}
	}

	if packed != "" {
		// The trainer prefers a packed corpus over text, so the counter must
		// too, or the two would disagree about what "the data" even is.
		info, err := corpus.PackedHeader(packed)
		if err != nil {
			return nil, err
		}
		if info.Name != v.Name {
			return nil, fmt.Errorf("%s holds variant %q but --variant is %q", packed, info.Name, v.Name)
		}
		return &row{
			Variant: v.Name, Source: srcPacked, FileCount: 1,
			// Neither games nor malformed lines survive packing: the format has
			// no game field, and the packer dropped bad lines before writing.
			// Both are "unknown", which is not the same as zero.
			Games:     -1,
			Malformed: -1,
			Lines:     info.Stats.Lines,
			Unique:    info.Stats.Unique,
			TBSkip:    info.Stats.TBSkip,
			Exact:     true, // whatever the packer counted, it counted exactly
			PerFile: []fileRow{{
				Path: packed, Games: -1, Malformed: -1, Lines: info.Stats.Lines,
				Unique: info.Stats.Unique, TBSkip: info.Stats.TBSkip,
			}},
		}, nil
	}

	o.Globs = globs
	res, err := corpus.Count(v, o)
	if err != nil {
		return nil, err
	}
	out := &row{
		Variant: v.Name, Source: srcText, FileCount: res.Files,
		Games: res.Games, Lines: res.Lines, Unique: res.Unique,
		Malformed: res.Malformed, TBSkip: res.TBSkip, NoGameID: res.NoGameID,
		Exact: res.Exact,
	}
	for _, fc := range res.PerFile {
		fr := fileRow{
			Path: fc.Path, Games: fc.Games, Lines: fc.Lines, Unique: fc.Unique,
			Malformed: fc.Malformed, TBSkip: fc.TBSkip,
		}
		if fc.Err != nil {
			fr.Error = fc.Err.Error()
		}
		out.PerFile = append(out.PerFile, fr)
	}
	return out, nil
}

// total sums the rows. UNIQUE is a sum of per-variant uniques, NOT a
// cross-variant dedup: the key spaces are different geometries, so a shared key
// between two variants would be meaningless.
func total(rows []row, exact bool) row {
	t := row{Variant: "TOTAL", Exact: exact}
	unknownGames, knownGames := false, false
	unknownMal, knownMal := false, false
	for _, r := range rows {
		t.FileCount += r.FileCount
		t.Lines += r.Lines
		t.TBSkip += r.TBSkip
		t.NoGameID += r.NoGameID
		// A -1 means "this source cannot answer". Summing the answerable rows
		// and flagging the rest beats printing a total that quietly treats
		// unknown as zero.
		if r.Games < 0 {
			unknownGames = true
		} else {
			knownGames = true
			t.Games += r.Games
		}
		if r.Malformed < 0 {
			unknownMal = true
		} else {
			knownMal = true
			t.Malformed += r.Malformed
		}
		if r.Unique < 0 {
			t.Unique = -1
		} else if t.Unique >= 0 {
			t.Unique += r.Unique
		}
	}
	if unknownGames && !knownGames {
		t.Games = -1
	}
	if unknownMal && !knownMal {
		t.Malformed = -1
	}
	return t
}

const tableFmt = "%-14s %-13s %5s %12s %14s %14s %6s %11s %11s\n"

func printTable(rep report, perFile, unique, exact bool) {
	fmt.Printf(tableFmt, "VARIANT", "SOURCE", "FILES", "GAMES", "LINES",
		"UNIQUE", "DEDUP", "MALFORMED", "TB-SKIP")
	for _, r := range rep.Variants {
		printRow(r)
	}
	printRow(rep.Total)

	fmt.Println()
	fmt.Println("MALFORMED lines are rejected BEFORE the loader counts them, so they are NOT in")
	fmt.Println("LINES. TB-SKIP lines ARE in LINES but are excluded from UNIQUE - they are two")
	fmt.Println("different things.")
	fmt.Println("GAMES is distinct gameids within each file, summed: gameid restarts at 0 in")
	fmt.Println("every worker file, so it is unique only within one file.")
	// Only worth saying when a text corpus was actually scanned: a packed row's
	// UNIQUE comes from the header, which the packer counted exactly.
	if unique && !exact && anyText(rep.Variants) {
		fmt.Println("UNIQUE dedups on a 64-bit hash of the position key (expected miscount below")
		fmt.Println("1e-4 even at 50M positions); --exact-unique uses the full key at ~5x the memory.")
	}
	if !unique {
		fmt.Println("UNIQUE and DEDUP are \"-\" because --unique=false was given.")
	}
	if anyPacked(rep.Variants) {
		fmt.Println("packed(.ntc) rows come from the file header, which carries no game count -")
		fmt.Println("hence \"-\" under GAMES. Count the .txt files with --data to get games.")
	}
	if rep.Total.NoGameID > 0 {
		fmt.Printf("%s accepted lines carry no gameid column (legacy 3-column data); those games\n",
			comma(rep.Total.NoGameID))
		fmt.Println("cannot be counted and are missing from GAMES.")
	}
	fmt.Println("UNIQUE in the TOTAL row is a sum of per-variant uniques, not a cross-variant")
	fmt.Println("dedup (the key spaces are different geometries).")
	if len(rep.NoData) > 0 {
		fmt.Printf("no data: %s\n", strings.Join(rep.NoData, " "))
	}

	if perFile {
		for _, r := range rep.Variants {
			fmt.Println()
			fmt.Printf("%s  (per-file UNIQUE is within that file only; it does not sum to %s)\n",
				r.Variant, dashOrComma(r.Unique))
			fmt.Printf("  %-52s %8s %10s %10s %10s %10s\n",
				"FILE", "GAMES", "LINES", "UNIQUE", "MALFORMED", "TB-SKIP")
			for _, fr := range r.PerFile {
				name := filepath.Base(fr.Path)
				// Chess pools two folders, so the folder is part of a file's
				// identity: two files can share a basename.
				if dir := filepath.Base(filepath.Dir(fr.Path)); dir != "." && dir != string(filepath.Separator) {
					name = dir + string(filepath.Separator) + name
				}
				fmt.Printf("  %-52s %8s %10s %10s %10s %10s\n", name,
					dashOrComma(fr.Games), comma(fr.Lines), dashOrComma(fr.Unique),
					comma(fr.Malformed), comma(fr.TBSkip))
				if fr.Error != "" {
					fmt.Printf("  %-52s READ FAILED: %s\n", "", fr.Error)
				}
			}
		}
	}

	if len(rep.Errors) > 0 {
		fmt.Println()
		fmt.Printf("%d file(s) could not be read; the counts above are incomplete:\n", len(rep.Errors))
		for _, e := range rep.Errors {
			fmt.Printf("  %s\n", e)
		}
	}
}

func printRow(r row) {
	dedup := "-"
	if r.Unique > 0 && r.Lines > 0 {
		dedup = fmt.Sprintf("%.1fx", float64(r.Lines)/float64(r.Unique))
	}
	fmt.Printf(tableFmt, r.Variant, r.Source, fmt.Sprint(r.FileCount),
		dashOrComma(r.Games), comma(r.Lines), dashOrComma(r.Unique), dedup,
		dashOrComma(r.Malformed), comma(r.TBSkip))
}

func anyPacked(rows []row) bool {
	for _, r := range rows {
		if r.Source == srcPacked {
			return true
		}
	}
	return false
}

func anyText(rows []row) bool {
	for _, r := range rows {
		if r.Source == srcText {
			return true
		}
	}
	return false
}

// searchedNote lists the roots that were looked in, so "it found nothing" is
// actionable. Note that Roots() includes the EXE's own directory, which is why
// running the binary from elsewhere can still find the kit's own data.
func searchedNote() string {
	var b strings.Builder
	b.WriteString("  Looked in:\n")
	for _, r := range datadir.Roots() {
		fmt.Fprintf(&b, "    %s\n", r)
	}
	b.WriteString("  Pass --data \"<glob>\" or --corpus <file.ntc> to point at a corpus.\n")
	return b.String()
}

func emitJSON(rep report) {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(rep); err != nil {
		die(1, "encoding JSON: %v", err)
	}
}

func dashOrComma(n int64) string {
	if n < 0 {
		return "-"
	}
	return comma(n)
}

// comma groups an integer in threes. This tool exists to be READ, and the
// count.ps1 it replaces formatted with {0:N0}.
func comma(n int64) string {
	neg := n < 0
	if neg {
		n = -n
	}
	s := fmt.Sprint(n)
	var out []byte
	for i, c := range []byte(s) {
		if i > 0 && (len(s)-i)%3 == 0 {
			out = append(out, ',')
		}
		out = append(out, c)
	}
	if neg {
		return "-" + string(out)
	}
	return string(out)
}

// die writes to STDERR and exits. Exit codes: 1 = a file could not be read,
// 2 = usage, 5 = no data anywhere. 3 and 4 are reserved: they already mean "no
// GPU" and "accumulator overflow" in the trainer binary, and two binaries in one
// kit must not give one code two meanings.
func die(code int, f string, a ...any) {
	fmt.Fprintf(os.Stderr, "count: "+f+"\n", a...)
	os.Exit(code)
}
