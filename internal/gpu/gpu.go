// Package gpu is the optional CUDA backend for the minibatch trainer.
//
// Go has no native GPU path, so the kernels are CUDA C and reach Go through
// cgo. That would normally destroy cross-compilation - but it does not here,
// because the two builds are split along the line where cross-compilation
// actually matters:
//
//	CGO_ENABLED=0 go build              -> pure Go, cross-compiles from Windows
//	                                       to linux/amd64, --gpu=1 exits 3
//	nvcc -c kernels.cu && go build -tags cuda
//	                                    -> built ON Colab, natively, where
//	                                       cross-compilation was never needed
//
// So cgo only ever appears in a build that happens on the machine it runs on.
//
// --gpu=1 on a non-CUDA binary FAILS LOUDLY rather than falling back. Silent
// fallback is the exact failure mode this project's build guards exist to
// prevent: a run that quietly did something other than what was asked produces
// a net nobody can account for. --gpu=auto opts into fallback explicitly.
package gpu

import "errors"

// ErrNoCUDA is returned by Open on a binary built without the cuda tag.
var ErrNoCUDA = errors.New(
	"this binary was built WITHOUT CUDA support, so --gpu=1 cannot be honoured.\n" +
		"  On Google Colab:  bash build_colab.sh    then run ./trainer_colab\n" +
		"                    (needs Runtime > Change runtime type > T4 GPU)\n" +
		"  On this machine:  drop --gpu, or use --gpu=auto to fall back to CPU,\n" +
		"                    or --algo=minibatch --threads=N which gets most of\n" +
		"                    the speedup without any CUDA at all")

// Corpus is the device-resident training set. Everything is uploaded once and
// stays on the GPU for the whole run - at ~12M positions this is roughly 500 MB
// and there is no per-step PCIe traffic, which is the entire reason the GPU
// path is worth having.
// Samples are packed TRAIN FIRST, then VALID, so the device can address the
// holdout as one contiguous range starting at TrainCount.
type Corpus struct {
	Pool   []uint16 // flat feature pool
	Off    []int32  // per-sample start into Pool
	End    []int32  // per-sample end
	Anchor []int16
	Bucket []uint8
	Y      []float32

	TrainCount int
	ValidCount int
}

// Weights is the parameter set moved between host and device.
type Weights struct {
	W1, B1, W2 []float32
}

// Params are the fixed shapes and hyperparameters of a run.
type Params struct {
	H, NumFeat, Buckets int
	Mirror              []uint16
	K                   float32
	Batch               int
	Opt                 string // "adam" or "adagrad"
}

// Device is one CUDA context with the corpus and weights resident.
type Device interface {
	Name() string
	Upload(c *Corpus, w *Weights, p *Params) error
	// RunEpoch consumes perm in order, in batches of p.Batch, applying one
	// optimizer step per batch. It must implement exactly the computation
	// documented on train.Batch.accumulate.
	RunEpoch(perm []int32, lr float32) error
	Download(w *Weights) error
	ValidMSE(c *Corpus) (float64, error)
	Close() error
}
