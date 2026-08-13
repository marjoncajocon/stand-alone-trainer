package corpus

// Corpus inventory: how many games and positions a set of selfplay files holds.
//
// This lives in package corpus rather than in cmd/count for one reason that
// matters more than tidiness: Count and Load can then be asserted EQUAL on the
// same input (count_test.go's TestCountMatchesLoad), and both go through the
// same draughtsAccept/chessAccept predicates. An inventory that disagreed with
// what training actually reads would be worse than no inventory.
//
// It replaces the engine kits' count.ps1, which counted games by grepping
// "done:" in the paired .log and therefore reported 0 games whenever the .log
// was missing. Games here come from the data itself.

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"

	"nnuetrainer/internal/chess"
	"nnuetrainer/internal/variant"
)

// CountOptions control a counting pass.
//
// Deliberately NOT corpus.Options: a count has no holdout, no lambda and no
// group key, and offering those knobs would imply they change the numbers.
//
// There is also deliberately no MaxLines. Load's version breaks out of a file
// mid-scan (corpus.go's `if o.MaxLines > 0 && nLines >= ...`), which under
// per-file parallel workers would make the per-file table depend on goroutine
// scheduling. Nothing here is slow enough to need it.
type CountOptions struct {
	Globs   []string
	Threads int  // 0 = min(NumCPU, len(files))
	Unique  bool // build the unique-position set (the only memory-hungry part)
	Exact   bool // with Unique: exact pack keys instead of a 64-bit hash
	PerFile bool // retain the per-file rows
}

// FileCount is one input file's contribution.
//
// Unique is unique WITHIN THIS FILE, so the per-file column does not sum to
// CountResult.Unique -- the same board appearing in two files is one unique
// position overall but one in each file's own tally.
type FileCount struct {
	Path      string
	Lines     int64 // what Load counts: post-accept, TB-skips INCLUDED
	Malformed int64 // rejected BEFORE Load's nLines++, so NOT in Lines
	TBSkip    int64 // in Lines, but excluded from Unique
	NoGameID  int64 // accepted lines with no gameid column (legacy 3-column data)
	Games     int64 // distinct gameids in this file
	Unique    int64 // distinct positions in this file (-1 if not counted)
	Err       error // this file only; the pass continues
}

// CountResult is the whole inventory for one variant.
type CountResult struct {
	V         *variant.Variant
	Files     int
	Lines     int64
	Malformed int64
	TBSkip    int64
	NoGameID  int64
	Games     int64
	Unique    int64 // -1 when CountOptions.Unique was false
	Exact     bool
	PerFile   []FileCount // sorted glob order; nil unless CountOptions.PerFile
	Errs      int
}

// Dedup is Lines/Unique, or 0 when Unique was not counted.
func (r *CountResult) Dedup() float64 {
	if r.Unique <= 0 {
		return 0
	}
	return float64(r.Lines) / float64(r.Unique)
}

// keyLen is how many bytes of a pack key a variant actually fills. The rest of
// the [maxPackBytes]byte array is a constant zero tail, so hashing only the
// prefix changes nothing and saves 13-18 bytes per line.
func keyLen(v *variant.Variant) int {
	if v.IsChess() {
		return chess.PackKeyLen
	}
	return v.Cells / 2
}

// mix64 is splitmix64's finalizer. FNV-1a has weak avalanche on short
// fixed-length inputs, and this removes any doubt about clustering for the price
// of three multiplies.
func mix64(x uint64) uint64 {
	x ^= x >> 30
	x *= 0xbf58476d1ce4e5b9
	x ^= x >> 27
	x *= 0x94d049bb133111eb
	x ^= x >> 31
	return x
}

// hashKey is FNV-1a-64 over the used prefix of a pack key, finalized.
//
// It must NOT be hash/maphash: that is seeded randomly per process, so two runs
// of the counter on the same corpus could print different UNIQUE values. The
// entire point of this tool is a number you can diff across runs, machines and
// operating systems, so the hash is fixed by hand.
func hashKey(k []byte) uint64 {
	h := uint64(14695981039346656037)
	for i := 0; i < len(k); i++ {
		h = (h ^ uint64(k[i])) * 1099511628211
	}
	return mix64(h)
}

// uniqSet is the two dedup strategies behind one interface.
//
// Hashed is the default. Memory per live entry is ~12 B for map[uint64]struct{}
// against ~64 B for the exact map[[50]byte]struct{} the loader uses -- 123 MB
// vs 640 MB at 10M unique positions, and roughly double each during map growth.
// Expected collisions are n(n-1)/2 / 2^64: 2.7e-6 at 10M keys, 6.8e-5 at 50M.
// The exact map costs 5x the memory to avoid an error nobody can observe, so it
// is opt-in (--exact-unique) and doubles as the hashed path's test oracle.
type uniqSet interface {
	add(k []byte)
	merge(other uniqSet)
	len() int64
}

type hashSet map[uint64]struct{}

func (s hashSet) add(k []byte) { s[hashKey(k)] = struct{}{} }
func (s hashSet) merge(other uniqSet) {
	for h := range other.(hashSet) {
		s[h] = struct{}{}
	}
}
func (s hashSet) len() int64 { return int64(len(s)) }

type exactSet map[[maxPackBytes]byte]struct{}

func (s exactSet) add(k []byte) {
	var a [maxPackBytes]byte
	copy(a[:], k)
	s[a] = struct{}{}
}
func (s exactSet) merge(other uniqSet) {
	for a := range other.(exactSet) {
		s[a] = struct{}{}
	}
}
func (s exactSet) len() int64 { return int64(len(s)) }

func newUniqSet(exact bool) uniqSet {
	if exact {
		return exactSet{}
	}
	return hashSet{}
}

// Count tallies games and positions across every file matching o.Globs.
func Count(v *variant.Variant, o CountOptions) (*CountResult, error) {
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
	// patterns produce one deterministic order, exactly as Load does.
	sort.Strings(files)

	threads := o.Threads
	if threads <= 0 {
		threads = runtime.NumCPU()
	}
	// The unit of work is a whole file, so more workers than files is waste.
	if threads > len(files) {
		threads = len(files)
	}

	res := &CountResult{V: v, Files: len(files), Exact: o.Exact, Unique: -1}

	// results is preallocated and indexed by the file's position in the sorted
	// list, so workers never append and never need a lock or an ordering pass:
	// the printed output is a function of sorted file order regardless of which
	// goroutine finished when.
	results := make([]FileCount, len(files))

	var global uniqSet
	merged := make(chan struct{})
	locals := make(chan uniqSet, 2)
	if o.Unique {
		global = newUniqSet(o.Exact)
		// A single merger goroutine owns the global set. Alternatives rejected:
		// merging every worker's set at the end peaks at 2x the global set, and
		// a sharded global map with per-shard mutexes would take a lock per
		// LINE. This way there is no synchronisation on the hot path at all,
		// and a merge overlaps other workers' scanning.
		go func() {
			for s := range locals {
				global.merge(s)
			}
			close(merged)
		}()
	} else {
		close(locals)
		close(merged)
	}

	jobs := make(chan int)
	done := make(chan struct{})
	for w := 0; w < threads; w++ {
		go func() {
			// Per-worker scratch, reused across every file this worker gets.
			fbuf := make([][]byte, 0, fieldCap(v))
			var pos chess.Position
			for i := range jobs {
				fc, local := countFile(v, files[i], o, fbuf, &pos)
				results[i] = fc
				if local != nil {
					locals <- local
				}
			}
			done <- struct{}{}
		}()
	}
	for i := range files {
		jobs <- i
	}
	close(jobs)
	for w := 0; w < threads; w++ {
		<-done
	}
	if o.Unique {
		close(locals)
	}
	<-merged

	for i := range results {
		fc := &results[i]
		if fc.Err != nil {
			res.Errs++
		}
		res.Lines += fc.Lines
		res.Malformed += fc.Malformed
		res.TBSkip += fc.TBSkip
		res.NoGameID += fc.NoGameID
		res.Games += fc.Games
	}
	if o.Unique {
		res.Unique = global.len()
	}
	if o.PerFile {
		res.PerFile = results
	}
	return res, nil
}

// fieldCap is how many whitespace fields a line can carry: chess lines are
// <fen(6)> <result> <gameid> <eval>, draughts are <board> <stm> <result>
// <gameid> <eval>. Matches Load's nFields exactly.
func fieldCap(v *variant.Variant) int {
	if v.IsChess() {
		return 9
	}
	return 5
}

// countFile scans one file. It returns the file's tallies and, when unique
// counting is on, the file's own key set for the merger to union.
func countFile(v *variant.Variant, path string, o CountOptions,
	fbuf [][]byte, pos *chess.Position) (FileCount, uniqSet) {

	fc := FileCount{Path: path, Unique: -1}
	fh, err := os.Open(path)
	if err != nil {
		// Not fatal: a live selfplay run is a normal thing to count, and a
		// partial inventory beats no inventory. cmd/count exits 1 and names
		// the file after printing the table.
		fc.Err = err
		return fc, nil
	}
	defer fh.Close()

	sc := bufio.NewScanner(fh)
	// This buffer size MUST match Load's (corpus.go's sc.Buffer call). With the
	// default 64 KB a long line would be silently dropped by one tool and
	// counted by the other, and the inventory would stop matching training.
	sc.Buffer(make([]byte, 0, 1<<20), 1<<20)

	var local uniqSet
	if o.Unique {
		local = newUniqSet(o.Exact)
	}
	// gameids restart at 0 in every worker file (stand-alone-selfplay writes one
	// file per worker process), so the scope is ONE file and the totals sum with
	// no cross-file dedup.
	gids := map[string]struct{}{}
	isChess := v.IsChess()

	// noteGid records one accepted line's game. The map LOOKUP with a
	// string([]byte) conversion is allocation-free (the compiler special-cases
	// it), so only the miss path allocates -- once per distinct game, not once
	// per line. Legacy 3-column data has no gameid at all; counting those lines
	// separately is what keeps GAMES from silently undercounting.
	noteGid := func(tok []byte, have bool) {
		if !have {
			fc.NoGameID++
			return
		}
		if _, seen := gids[string(tok)]; !seen {
			gids[string(tok)] = struct{}{}
		}
	}

	for sc.Scan() {
		line := sc.Bytes()
		parts := fields(line, fbuf)

		var key [maxPackBytes]byte
		var gidTok []byte
		var haveGid bool

		if isChess {
			if _, ok := chessAccept(line, parts, pos); !ok {
				fc.Malformed++
				continue
			}
			fc.Lines++
			gidTok, haveGid = parts[7], true
			pos.PackKey(key[:])
		} else {
			if !draughtsAccept(parts, v.Cells) {
				fc.Malformed++
				continue
			}
			fc.Lines++
			board := parts[0]
			if len(parts) >= 4 {
				gidTok, haveGid = parts[3], true
			}

			// ONE DELIBERATE DIVERGENCE FROM Load: Load reads the draughts
			// gameid AFTER its TB-skip `continue`, so a game whose every line is
			// TB territory contributes no holdout group. The gameid is counted
			// BEFORE the skip here, because "how many games are in this corpus"
			// is a question about the data, not about a training filter -- and it
			// is what makes GAMES comparable to the .log "done:" count.
			if v.TBSkip(v.Counts(board)) {
				fc.TBSkip++
				noteGid(gidTok, haveGid)
				continue
			}
			key = packKey(board, v.Cells)
		}

		noteGid(gidTok, haveGid)
		if local != nil {
			local.add(key[:keyLen(v)])
		}
	}
	if err := sc.Err(); err != nil {
		fc.Err = err
	}
	fc.Games = int64(len(gids))
	if local != nil {
		fc.Unique = local.len()
	}
	return fc, local
}
