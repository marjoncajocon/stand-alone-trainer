//go:build !cuda

package gpu

// Available reports whether this binary can use a GPU. Always false here: this
// file is compiled when the `cuda` build tag is absent, i.e. in every pure-Go
// build, which is every build that cross-compiles.
func Available() bool { return false }

// Backend names the compiled-in backend, for --version and startup banners.
func Backend() string { return "none (pure Go, CGO_ENABLED=0)" }

// Open always fails here, with an error that says how to get a CUDA build.
func Open(ordinal int) (Device, error) { return nil, ErrNoCUDA }
