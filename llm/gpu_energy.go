package llm

import (
	"log/slog"
	"sync"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/energy"
)

// runningGPUWork counts the work running on each GPU, keyed by PCI bus ID,
// so that board energy is only reported for a request that had its GPUs to
// itself. Every llama-server runner records its model load and each call
// that runs the model here, metered or not.
var runningGPUWork = struct {
	sync.Mutex
	running map[string]int
	started map[string]uint64
}{
	running: make(map[string]int),
	started: make(map[string]uint64),
}

func gpuPCIIDs(gpus []ml.DeviceInfo) []string {
	var ids []string
	for _, gpu := range gpus {
		if gpu.PCIID != "" {
			ids = append(ids, gpu.PCIID)
		}
	}
	return ids
}

// openNVML is replaced in tests.
var openNVML = energy.Open

// openEnergyMeter returns a meter over the GPUs, or nil when none of the
// model's layers are on them or the GPUs are not all NVIDIA GPUs with a known
// PCI bus ID and an energy counter.
func openEnergyMeter(gpus []ml.DeviceInfo, gpuLayers uint64) energy.Meter {
	if len(gpus) == 0 || gpuLayers == 0 {
		return nil
	}

	ids := make([]string, 0, len(gpus))
	for _, gpu := range gpus {
		if gpu.Library != "CUDA" || gpu.PCIID == "" {
			return nil
		}
		ids = append(ids, gpu.PCIID)
	}

	meter, err := openNVML(ids)
	if err != nil {
		slog.Debug("GPU energy will not be reported", "error", err)
		return nil
	}
	return meter
}

// gpuWork marks work as running on a set of GPUs.
type gpuWork struct {
	gpus    []string
	started []uint64
	alone   bool
	done    bool
}

func startGPUWork(gpus []string) *gpuWork {
	w := &gpuWork{gpus: gpus, alone: true}
	runningGPUWork.Lock()
	defer runningGPUWork.Unlock()
	for _, id := range gpus {
		runningGPUWork.running[id]++
		runningGPUWork.started[id]++
		if runningGPUWork.running[id] > 1 {
			w.alone = false
		}
		w.started = append(w.started, runningGPUWork.started[id])
	}
	return w
}

// finish reports whether the work had its GPUs to itself from start to
// finish. It is safe to call more than once, and only the first call counts.
func (w *gpuWork) finish() bool {
	if w == nil {
		return false
	}

	runningGPUWork.Lock()
	defer runningGPUWork.Unlock()
	if w.done {
		return false
	}
	w.done = true

	alone := w.alone
	for i, id := range w.gpus {
		if runningGPUWork.started[id] != w.started[i] {
			alone = false
		}
		runningGPUWork.running[id]--
		if runningGPUWork.running[id] == 0 {
			delete(runningGPUWork.running, id)
			delete(runningGPUWork.started, id)
		}
	}
	return alone
}

// energyWindow measures one request. It belongs to the goroutine that
// handles the request.
type energyWindow struct {
	meter energy.Meter
	work  *gpuWork
	begin uint64
	err   error
	ended bool
}

func beginEnergyWindow(meter energy.Meter, gpus []string) *energyWindow {
	w := &energyWindow{meter: meter, work: startGPUWork(gpus)}
	if meter != nil {
		w.begin, w.err = meter.Read()
	}
	return w
}

// end returns the joules the GPUs drew since the window began. It returns 0
// when there is no meter, the energy can't be read, the counter didn't move,
// or other work ran on the same GPUs meanwhile. Only the first call counts.
func (w *energyWindow) end() float64 {
	if w.ended {
		return 0
	}
	w.ended = true

	if w.meter == nil {
		w.work.finish()
		return 0
	}

	mj, err := w.meter.Read()
	alone := w.work.finish()
	if !alone || w.err != nil || err != nil || mj < w.begin {
		return 0
	}
	return float64(mj-w.begin) / 1000
}
