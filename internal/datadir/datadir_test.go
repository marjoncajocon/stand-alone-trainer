package datadir

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"nnuetrainer/internal/variant"
)

func get(t *testing.T, name string) *variant.Variant {
	t.Helper()
	v, err := variant.Get(name)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

// TestFolders pins the one asymmetry in the layout: chess pools two folders into
// one net, every draughts variant reads exactly its own.
func TestFolders(t *testing.T) {
	if got, want := Folders(get(t, "chess")), []string{"chess", "chess960"}; !reflect.DeepEqual(got, want) {
		t.Errorf("chess folders = %v, want %v", got, want)
	}
	if got, want := Folders(get(t, "filipino")), []string{"filipino"}; !reflect.DeepEqual(got, want) {
		t.Errorf("filipino folders = %v, want %v", got, want)
	}
}

// TestDiscoverPrefersPacked is why discoverIn is separated from Discover: the
// preference decides whether the trainer trains on a frozen split and whether
// the counter reports header numbers or scans text, so it has to be testable.
func TestDiscoverPrefersPacked(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "filipino"), 0o755); err != nil {
		t.Fatal(err)
	}
	txt := filepath.Join(root, "filipino", "a.txt")
	if err := os.WriteFile(txt, []byte("x\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	v := get(t, "filipino")
	// Text only, first.
	pk, globs, searched := discoverIn([]string{root}, v)
	if pk != "" || len(globs) != 1 || searched != "" {
		t.Fatalf("text-only: got (%q, %v, %q)", pk, globs, searched)
	}

	ntc := filepath.Join(root, "filipino.ntc")
	if err := os.WriteFile(ntc, []byte("NTC2"), 0o644); err != nil {
		t.Fatal(err)
	}
	pk, globs, _ = discoverIn([]string{root}, v)
	if pk != ntc {
		t.Errorf("packed = %q, want %q", pk, ntc)
	}
	if globs != nil {
		t.Errorf("globs = %v, want nil when a packed corpus is present", globs)
	}
}

// TestDiscoverSearchedList checks the not-found path names where it looked; the
// error message is the whole value of that return.
func TestDiscoverSearchedList(t *testing.T) {
	a, b := t.TempDir(), t.TempDir()
	pk, globs, searched := discoverIn([]string{a, b}, get(t, "filipino"))
	if pk != "" || globs != nil {
		t.Fatalf("empty roots found something: (%q, %v)", pk, globs)
	}
	for _, want := range []string{a, b} {
		if !strings.Contains(searched, want) {
			t.Errorf("searched list %q does not mention %q", searched, want)
		}
	}
}

// TestRootsAreAbsoluteAndUnique guards the deduplication: running the binary
// from its own directory makes exe-dir/data and cwd/data the same path, and a
// duplicate root would make the not-found message list it twice.
func TestRootsAreAbsoluteAndUnique(t *testing.T) {
	seen := map[string]bool{}
	for _, r := range Roots() {
		if !filepath.IsAbs(r) {
			t.Errorf("root %q is not absolute", r)
		}
		if seen[r] {
			t.Errorf("root %q listed twice", r)
		}
		seen[r] = true
	}
	if len(seen) == 0 {
		t.Error("Roots() is empty")
	}
}
