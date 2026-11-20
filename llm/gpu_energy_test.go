package llm

import (
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"

	"golang.org/x/sync/semaphore"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/energy"
)

// fakeMeter adds step millijoules on every read.
type fakeMeter struct {
	mu     sync.Mutex
	mj     uint64
	step   uint64
	reads  int
	err    error
	closed bool
}

func (m *fakeMeter) Read() (uint64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.reads++
	if m.err != nil {
		return 0, m.err
	}
	m.mj += m.step
	return m.mj, nil
}

func (m *fakeMeter) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed = true
	return nil
}

const (
	gpu0 = "0000:01:00.0"
	gpu1 = "0000:02:00.0"
)

func cudaGPU(pci string) ml.DeviceInfo {
	return ml.DeviceInfo{DeviceID: ml.DeviceID{ID: "0", Library: "CUDA"}, PCIID: pci}
}

// isolateGPUWork starts the test with no GPU work recorded and fails it
// if the test leaves any behind.
func isolateGPUWork(t *testing.T) {
	t.Helper()
	runningGPUWork.Lock()
	runningGPUWork.running = make(map[string]int)
	runningGPUWork.started = make(map[string]uint64)
	runningGPUWork.Unlock()

	t.Cleanup(func() {
		runningGPUWork.Lock()
		defer runningGPUWork.Unlock()
		if len(runningGPUWork.running) != 0 {
			t.Errorf("GPU work left running: %v", runningGPUWork.running)
		}
	})
}

func TestEnergyWindowReportsJoules(t *testing.T) {
	isolateGPUWork(t)
	meter := &fakeMeter{step: 1500}
	w := beginEnergyWindow(meter, []string{gpu0})
	if got := w.end(); got != 1.5 {
		t.Errorf("end() = %v, want 1.5", got)
	}
	if got := w.end(); got != 0 {
		t.Errorf("second end() = %v, want 0", got)
	}
	if meter.reads != 2 {
		t.Errorf("meter read %d times, want 2", meter.reads)
	}
}

func TestEnergyWindowNilMeter(t *testing.T) {
	isolateGPUWork(t)
	w := beginEnergyWindow(nil, []string{gpu0})
	if got := w.end(); got != 0 {
		t.Errorf("end() = %v, want 0", got)
	}
}

func TestEnergyWindowReadError(t *testing.T) {
	isolateGPUWork(t)
	meter := &fakeMeter{step: 1000}
	w := beginEnergyWindow(meter, []string{gpu0})
	meter.err = errors.New("counter unavailable")
	if got := w.end(); got != 0 {
		t.Errorf("end() = %v, want 0", got)
	}

	meter.err = nil
	if got := beginEnergyWindow(meter, []string{gpu0}).end(); got != 1 {
		t.Errorf("end() after a failed read = %v, want 1", got)
	}
}

func TestEnergyWindowCounterReset(t *testing.T) {
	isolateGPUWork(t)
	// The counter restarts from zero when the driver reloads.
	meter := &fakeMeter{mj: 5_000_000, step: 1000}
	w := beginEnergyWindow(meter, []string{gpu0})
	meter.mj = 0
	if got := w.end(); got != 0 {
		t.Errorf("end() = %v, want 0", got)
	}
}

func TestEnergyWindowSkipsOverlappingRequests(t *testing.T) {
	isolateGPUWork(t)
	meter := &fakeMeter{step: 1000}

	// b starts while a runs and ends first, c on another GPU overlaps both.
	a := beginEnergyWindow(meter, []string{gpu0})
	b := beginEnergyWindow(meter, []string{gpu0})
	c := beginEnergyWindow(meter, []string{gpu1})
	if got := b.end(); got != 0 {
		t.Errorf("b.end() = %v, want 0", got)
	}
	if got := a.end(); got != 0 {
		t.Errorf("a.end() = %v, want 0", got)
	}
	if got := c.end(); got == 0 {
		t.Error("c.end() = 0, want the energy of a request alone on its GPU")
	}

	// A request that starts after the others finished is alone again.
	if got := beginEnergyWindow(meter, []string{gpu0}).end(); got != 1 {
		t.Errorf("end() = %v, want 1", got)
	}
}

func TestEnergyWindowSpanningGPUs(t *testing.T) {
	isolateGPUWork(t)
	meter := &fakeMeter{step: 1000}

	a := beginEnergyWindow(meter, []string{gpu0, gpu1})
	b := beginEnergyWindow(meter, []string{gpu1})
	if got := b.end(); got != 0 {
		t.Errorf("b.end() = %v, want 0", got)
	}
	if got := a.end(); got != 0 {
		t.Errorf("a.end() = %v, want 0", got)
	}
}

func TestEnergyWindowUnmeteredWorkCounts(t *testing.T) {
	isolateGPUWork(t)
	meter := &fakeMeter{step: 1000}

	// Work from a runner without a meter, such as one also spread over a
	// GPU that has no energy counter.
	metered := beginEnergyWindow(meter, []string{gpu0})
	unmetered := beginEnergyWindow(nil, []string{gpu0, gpu1})
	unmetered.end()
	if got := metered.end(); got != 0 {
		t.Errorf("end() = %v, want 0", got)
	}
}

func TestEnergyWindowEndTwiceKeepsCount(t *testing.T) {
	isolateGPUWork(t)
	meter := &fakeMeter{step: 1000}

	a := beginEnergyWindow(meter, []string{gpu0})
	b := beginEnergyWindow(meter, []string{gpu0})
	a.end()
	a.end()

	// b is still running, so c is not alone.
	c := beginEnergyWindow(meter, []string{gpu0})
	if got := c.end(); got != 0 {
		t.Errorf("c.end() = %v, want 0", got)
	}
	b.end()
}

func TestGPUPCIIDs(t *testing.T) {
	gpus := []ml.DeviceInfo{
		{DeviceID: ml.DeviceID{ID: "0", Library: "CUDA"}, PCIID: gpu0},
		{DeviceID: ml.DeviceID{ID: "1", Library: "CUDA"}},
		{DeviceID: ml.DeviceID{ID: "0", Library: "ROCm"}, PCIID: gpu1},
	}
	if got := strings.Join(gpuPCIIDs(gpus), ","); got != gpu0+","+gpu1 {
		t.Errorf("gpuPCIIDs = %s, want %s,%s", got, gpu0, gpu1)
	}
}

func TestOpenEnergyMeter(t *testing.T) {
	rocm := ml.DeviceInfo{DeviceID: ml.DeviceID{ID: "0", Library: "ROCm"}, PCIID: gpu1}

	tests := []struct {
		name      string
		gpus      []ml.DeviceInfo
		gpuLayers uint64
		openErr   error
		wantOpen  []string
		wantMeter bool
	}{
		{name: "no gpus", gpuLayers: 29},
		{name: "cuda", gpus: []ml.DeviceInfo{cudaGPU(gpu0), cudaGPU(gpu1)}, gpuLayers: 29, wantOpen: []string{gpu0, gpu1}, wantMeter: true},
		{name: "partial offload", gpus: []ml.DeviceInfo{cudaGPU(gpu0)}, gpuLayers: 10, wantOpen: []string{gpu0}, wantMeter: true},
		{name: "no layers on the gpu", gpus: []ml.DeviceInfo{cudaGPU(gpu0)}},
		{name: "missing pci id", gpus: []ml.DeviceInfo{cudaGPU(gpu0), cudaGPU("")}, gpuLayers: 29},
		{name: "mixed vendors", gpus: []ml.DeviceInfo{cudaGPU(gpu0), rocm}, gpuLayers: 29},
		{name: "no counter", gpus: []ml.DeviceInfo{cudaGPU(gpu0)}, gpuLayers: 29, openErr: errors.New("not supported"), wantOpen: []string{gpu0}},
	}

	t.Cleanup(func() { openNVML = energy.Open })
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var opened []string
			openNVML = func(ids []string) (energy.Meter, error) {
				opened = ids
				if tt.openErr != nil {
					return nil, tt.openErr
				}
				return &fakeMeter{}, nil
			}

			meter := openEnergyMeter(tt.gpus, tt.gpuLayers)
			if strings.Join(opened, ",") != strings.Join(tt.wantOpen, ",") {
				t.Errorf("opened %v, want %v", opened, tt.wantOpen)
			}
			if (meter != nil) != tt.wantMeter {
				t.Errorf("meter = %v, want one: %v", meter, tt.wantMeter)
			}
		})
	}
}

func meteredRunner(t *testing.T, handler http.HandlerFunc, meter energy.Meter) *llamaServerRunner {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)

	port, err := strconv.Atoi(srv.URL[strings.LastIndex(srv.URL, ":")+1:])
	if err != nil {
		t.Fatal(err)
	}

	runner := &llamaServerRunner{
		port:    port,
		cmd:     fakeRunningCmd(),
		sem:     semaphore.NewWeighted(1),
		options: api.Options{Runner: api.Runner{NumCtx: 2048, NumGPU: -1}},
		gpus:    []ml.DeviceInfo{cudaGPU(gpu0)},
	}
	if meter != nil {
		runner.energyMeter = meter
		runner.meterOnce.Do(func() {})
	}
	return runner
}

// fakeNVML makes the runner open fakeMeters, and records each open.
func fakeNVML(t *testing.T, step uint64) *[]*fakeMeter {
	t.Helper()
	var opened []*fakeMeter
	openNVML = func([]string) (energy.Meter, error) {
		m := &fakeMeter{step: step}
		opened = append(opened, m)
		return m, nil
	}
	t.Cleanup(func() { openNVML = energy.Open })
	return &opened
}

func completionServer(t *testing.T) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			fmt.Fprint(w, `{"status":"ok"}`)
		case "/completion":
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprintln(w, `data: {"content":"","stop":true,"stop_type":"eos","timings":{"prompt_n":5,"prompt_ms":10,"predicted_n":2,"predicted_ms":20}}`)
		default:
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
	}
}

func complete(t *testing.T, runner *llamaServerRunner) CompletionResponse {
	t.Helper()
	var final CompletionResponse
	opts := api.DefaultOptions()
	err := runner.Completion(t.Context(), CompletionRequest{Prompt: "test prompt", Options: &opts}, func(cr CompletionResponse) {
		if cr.Done {
			final = cr
		}
	})
	if err != nil {
		t.Fatalf("Completion error: %v", err)
	}
	return final
}

func healthOK(t *testing.T) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		fmt.Fprint(w, `{"status":"ok"}`)
	}
}

func TestLlamaServerCompletionReportsGPUEnergy(t *testing.T) {
	isolateGPUWork(t)
	runner := meteredRunner(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			fmt.Fprint(w, `{"status":"ok"}`)
		case "/completion":
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprintln(w, `data: {"content":"Hi","stop":false}`)
			fmt.Fprintln(w, `data: {"content":"","stop":true,"stop_type":"eos","timings":{"prompt_n":5,"prompt_ms":10,"predicted_n":2,"predicted_ms":20}}`)
		default:
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
	}, &fakeMeter{step: 2500})

	var final CompletionResponse
	opts := api.DefaultOptions()
	err := runner.Completion(t.Context(), CompletionRequest{Prompt: "test prompt", Options: &opts}, func(cr CompletionResponse) {
		if cr.Done {
			final = cr
		} else if cr.GPUEnergy != 0 {
			t.Errorf("partial response has GPUEnergy %v", cr.GPUEnergy)
		}
	})
	if err != nil {
		t.Fatalf("Completion error: %v", err)
	}
	if final.GPUEnergy != 2.5 {
		t.Errorf("GPUEnergy = %v, want 2.5", final.GPUEnergy)
	}
}

func TestLlamaServerChatReportsGPUEnergy(t *testing.T) {
	isolateGPUWork(t)
	runner := meteredRunner(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			fmt.Fprint(w, `{"status":"ok"}`)
		case "/v1/chat/completions":
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprintln(w, `data: {"choices":[{"delta":{"content":"Hello"}}]}`)
			fmt.Fprintln(w, `data: {"choices":[{"delta":{},"finish_reason":"stop"}],"timings":{"prompt_n":5,"prompt_ms":10,"predicted_n":2,"predicted_ms":20}}`)
			fmt.Fprintln(w, `data: [DONE]`)
		default:
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
	}, &fakeMeter{step: 4000})

	var final ChatResponse
	opts := api.DefaultOptions()
	err := runner.Chat(t.Context(), ChatRequest{
		Messages: []api.Message{{Role: "user", Content: "test prompt"}},
		Options:  &opts,
	}, func(cr ChatResponse) {
		if cr.Done {
			final = cr
		}
	})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	if final.GPUEnergy != 4 {
		t.Errorf("GPUEnergy = %v, want 4", final.GPUEnergy)
	}
}

func TestLlamaServerFailedRequestEndsEnergyWindow(t *testing.T) {
	isolateGPUWork(t)
	runner := meteredRunner(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			fmt.Fprint(w, `{"status":"ok"}`)
		case "/completion", "/v1/chat/completions":
			http.Error(w, `{"error":{"message":"boom"}}`, http.StatusInternalServerError)
		default:
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
	}, &fakeMeter{step: 1000})

	opts := api.DefaultOptions()
	requests := map[string]func() error{
		"completion": func() error {
			return runner.Completion(t.Context(), CompletionRequest{Prompt: "test prompt", Options: &opts}, func(CompletionResponse) {})
		},
		"chat": func() error {
			return runner.Chat(t.Context(), ChatRequest{
				Messages: []api.Message{{Role: "user", Content: "test prompt"}},
				Options:  &opts,
			}, func(ChatResponse) {})
		},
	}
	for name, request := range requests {
		t.Run(name, func(t *testing.T) {
			if err := request(); err == nil {
				t.Fatal("expected an error")
			}
			if got := beginEnergyWindow(runner.energyMeter, []string{gpu0}).end(); got != 1 {
				t.Errorf("end() after a failed request = %v, want 1", got)
			}
		})
	}
}

func TestLlamaServerEmbeddingCountsAsGPUUse(t *testing.T) {
	isolateGPUWork(t)
	meter := &fakeMeter{step: 1000}
	runner := meteredRunner(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			fmt.Fprint(w, `{"status":"ok"}`)
		case "/v1/embeddings":
			fmt.Fprint(w, `{"data":[{"embedding":[0.1,0.2]}],"usage":{"prompt_tokens":2}}`)
		default:
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
	}, meter)

	generation := beginEnergyWindow(meter, []string{gpu0})
	if _, _, err := runner.Embedding(t.Context(), "hello", nil); err != nil {
		t.Fatalf("Embedding error: %v", err)
	}
	if got := generation.end(); got != 0 {
		t.Errorf("generation overlapping an embedding reported %v J, want 0", got)
	}
	if meter.reads != 2 {
		t.Errorf("meter read %d times, want 2: the embedding itself should not read it", meter.reads)
	}
}

// Speech and decision calls report no energy, but a generation that
// overlaps one of them must not claim the GPU's whole draw.
func TestLlamaServerSpeechAndSystemOneCountAsGPUUse(t *testing.T) {
	overlaps := func(t *testing.T, runner *llamaServerRunner, call func() error) {
		t.Helper()
		isolateGPUWork(t)
		runner.gpus = []ml.DeviceInfo{cudaGPU(gpu0)}
		generation := beginEnergyWindow(&fakeMeter{step: 1000}, []string{gpu0})
		if err := call(); err != nil {
			t.Fatal(err)
		}
		if got := generation.end(); got != 0 {
			t.Errorf("generation overlapping the call reported %v J, want 0", got)
		}
	}

	t.Run("speech", func(t *testing.T) {
		runner := newSpeechRunner(t, []string{"mmproj.gguf"}, func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", ttsContentType)
			w.Write(float32LE(0, 0.5))
		})
		overlaps(t, runner, func() error {
			req := lfm3Speech(t, SpeechRequest{Input: "hello there.", Voice: "male-french"})
			return runner.Speech(t.Context(), req, func([]byte) error { return nil })
		})
	})

	t.Run("systemone", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			fmt.Fprint(w, `{"answers":{},"usage":{}}`)
		}))
		t.Cleanup(srv.Close)
		runner := &llamaServerRunner{port: srv.Listener.Addr().(*net.TCPAddr).Port, client: srv.Client(), cmd: fakeRunningCmd(), sem: semaphore.NewWeighted(1)}
		overlaps(t, runner, func() error {
			_, _, err := runner.SystemOne(t.Context(), json.RawMessage(`"x"`), json.RawMessage(`{}`), nil)
			return err
		})
	})
}

func TestLlamaServerCloseClosesEnergyMeter(t *testing.T) {
	isolateGPUWork(t)
	meter := &fakeMeter{}
	runner := &llamaServerRunner{energyMeter: meter}
	if err := runner.Close(); err != nil {
		t.Fatal(err)
	}
	if !meter.closed {
		t.Error("energy meter was not closed")
	}
}

func TestLlamaServerLoadingCountsAsGPUUse(t *testing.T) {
	isolateGPUWork(t)
	meter := &fakeMeter{step: 1000}
	runner := meteredRunner(t, healthOK(t), meter)
	runner.loading = startGPUWork(runner.gpuIDs())

	during := beginEnergyWindow(meter, []string{gpu0})
	if _, err := runner.Load(t.Context(), ml.SystemInfo{}, nil, false); err != nil {
		t.Fatalf("Load error: %v", err)
	}
	if got := during.end(); got != 0 {
		t.Errorf("request overlapping a model load reported %v J, want 0", got)
	}
	if got := beginEnergyWindow(meter, []string{gpu0}).end(); got != 1 {
		t.Errorf("request after the load reported %v J, want 1", got)
	}
}

func TestLlamaServerCloseEndsUnfinishedLoad(t *testing.T) {
	isolateGPUWork(t)
	runner := &llamaServerRunner{gpus: []ml.DeviceInfo{cudaGPU(gpu0)}}
	runner.loading = startGPUWork(runner.gpuIDs())
	if err := runner.Close(); err != nil {
		t.Fatal(err)
	}
	if got := beginEnergyWindow(&fakeMeter{step: 1000}, []string{gpu0}).end(); got != 1 {
		t.Errorf("request after Close reported %v J, want 1", got)
	}
}

func TestLlamaServerLoadThenCloseFinishesLoadOnce(t *testing.T) {
	isolateGPUWork(t)
	meter := &fakeMeter{step: 1000}
	runner := meteredRunner(t, healthOK(t), meter)
	runner.loading = startGPUWork(runner.gpuIDs())

	request := beginEnergyWindow(meter, []string{gpu0})
	if _, err := runner.Load(t.Context(), ml.SystemInfo{}, nil, false); err != nil {
		t.Fatalf("Load error: %v", err)
	}
	if err := runner.Close(); err != nil {
		t.Fatal(err)
	}

	// request is still running, so a second one is not alone.
	if got := beginEnergyWindow(meter, []string{gpu0}).end(); got != 0 {
		t.Errorf("second request reported %v J, want 0", got)
	}
	request.end()
}

func TestLlamaServerOpensMeterAtLoad(t *testing.T) {
	isolateGPUWork(t)
	opened := fakeNVML(t, 3000)
	runner := meteredRunner(t, completionServer(t), nil)
	runner.gpuLayers = 29

	if _, err := runner.Load(t.Context(), ml.SystemInfo{}, nil, false); err != nil {
		t.Fatalf("Load error: %v", err)
	}
	if len(*opened) != 1 {
		t.Fatalf("NVML opened %d times by Load, want 1", len(*opened))
	}
	if got := complete(t, runner).GPUEnergy; got != 3 {
		t.Errorf("GPUEnergy = %v, want 3", got)
	}
	if len(*opened) != 1 {
		t.Errorf("NVML opened %d times, want 1", len(*opened))
	}

	if err := runner.Close(); err != nil {
		t.Fatal(err)
	}
	if !(*opened)[0].closed {
		t.Error("energy meter was not closed")
	}
}

// llama-server can fit every layer on the CPU even when the runner was
// given GPUs.
func TestLlamaServerNoGPULayersReportsNoEnergy(t *testing.T) {
	isolateGPUWork(t)
	opened := fakeNVML(t, 3000)
	runner := meteredRunner(t, completionServer(t), nil)

	if _, err := runner.Load(t.Context(), ml.SystemInfo{}, nil, false); err != nil {
		t.Fatalf("Load error: %v", err)
	}
	if got := complete(t, runner).GPUEnergy; got != 0 {
		t.Errorf("GPUEnergy = %v, want 0", got)
	}
	if len(*opened) != 0 {
		t.Errorf("NVML opened %d times, want 0", len(*opened))
	}
}

func TestLlamaServerCloseBeforeLoadOpensNoMeter(t *testing.T) {
	isolateGPUWork(t)
	opened := fakeNVML(t, 3000)
	runner := &llamaServerRunner{gpus: []ml.DeviceInfo{cudaGPU(gpu0)}, gpuLayers: 29}
	if err := runner.Close(); err != nil {
		t.Fatal(err)
	}
	if runner.meter() != nil || len(*opened) != 0 {
		t.Errorf("meter opened after Close (%d opens)", len(*opened))
	}
}
