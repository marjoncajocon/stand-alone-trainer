// Package datadir resolves where a corpus lives.
//
// This is shared by cmd/trainer and cmd/count on purpose: the counter must
// report on exactly the files the trainer would read, so "how much data do I
// have" and "what will training load" can never disagree because two copies of
// the search order drifted apart.
package datadir

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"nnuetrainer/internal/variant"
)

// Roots lists where a corpus is looked for, in priority order: beside the
// binary first (bin\data\<variant>, the same convention stand-alone-selfplay
// uses for --out-dir), then the kit root, then the current directory.
func Roots() []string {
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

// Folders lists the data/<folder> directories a variant reads.
//
// Chess reads TWO: standard and Chess960 pool into one net, because the eval is
// position-agnostic and the app ships a single net -- the same choice
// chess-cli's own kit makes (tools/trainer/README.txt:91-94). The folders stay
// separate on disk so the generator can keep them from mixing by accident
// (stand-alone-selfplay's SP_TOKEN_FOR), and the loader's holdout group key
// uses the folder name so a standard file and a 960 file with the same
// basename never share a group.
func Folders(v *variant.Variant) []string {
	if v.IsChess() {
		return []string{"chess", "chess960"}
	}
	return []string{v.Name}
}

// Discover finds a corpus for v without any flags. Returns (packedPath,
// textGlobs, searchedDescription); at most one of the first two is non-empty.
func Discover(v *variant.Variant) (string, []string, string) {
	return discoverIn(Roots(), v)
}

// discoverIn is Discover with the root list injected, which is the only way a
// test can exercise the search order: os.Executable and os.Getwd are not
// overridable.
func discoverIn(roots []string, v *variant.Variant) (string, []string, string) {
	var searched strings.Builder
	folders := Folders(v)
	for _, root := range roots {
		fmt.Fprintf(&searched, "    %s\n", root)
		// A packed corpus wins: faster to load and its split is frozen.
		pk := filepath.Join(root, v.Name+".ntc")
		if _, err := os.Stat(pk); err == nil {
			return pk, nil, ""
		}
		var globs []string
		for _, f := range folders {
			glob := filepath.Join(root, f, "*.txt")
			if m, err := filepath.Glob(glob); err == nil && len(m) > 0 {
				globs = append(globs, glob)
			}
		}
		if len(globs) > 0 {
			return "", globs, ""
		}
	}
	return "", nil, searched.String()
}
