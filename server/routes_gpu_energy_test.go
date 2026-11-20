package server

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
	"github.com/ollama/ollama/llm"
)

func TestGPUEnergyInResponses(t *testing.T) {
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "4096")
	t.Setenv("OLLAMA_GO_TEMPLATE", "")
	gin.SetMode(gin.TestMode)

	mock := mockRunner{
		CompletionResponse: llm.CompletionResponse{
			Content:      "ok",
			Done:         true,
			DoneReason:   llm.DoneReasonStop,
			EvalCount:    1,
			EvalDuration: time.Millisecond,
			GPUEnergy:    1.5,
		},
		ChatFn: func(_ context.Context, _ llm.ChatRequest, fn func(llm.ChatResponse)) error {
			fn(llm.ChatResponse{
				Message:      api.Message{Role: "assistant", Content: "ok"},
				Done:         true,
				DoneReason:   llm.DoneReasonStop,
				EvalCount:    1,
				EvalDuration: time.Millisecond,
				GPUEnergy:    2.5,
			})
			return nil
		},
	}
	s := newServerWithMockRunner(t, &mock)
	createMinimalGGUFModel(t, s, "go-template", nil, "{{ .Prompt }}", nil)
	createMinimalGGUFModel(t, s, "chat-template", gguftest.KV{
		"tokenizer.chat_template": "{{ messages[0]['content'] }}",
	}, "", nil)

	stream := false
	tests := []struct {
		name    string
		handler func(*gin.Context)
		body    any
		want    float64
	}{
		{
			name:    "generate",
			handler: s.GenerateHandler,
			body:    api.GenerateRequest{Model: "go-template", Prompt: "hello", Stream: &stream},
			want:    1.5,
		},
		{
			name:    "chat with a Go template",
			handler: s.ChatHandler,
			body:    api.ChatRequest{Model: "go-template", Messages: []api.Message{{Role: "user", Content: "hello"}}, Stream: &stream},
			want:    1.5,
		},
		{
			name:    "chat with a GGUF chat template",
			handler: s.ChatHandler,
			body:    api.ChatRequest{Model: "chat-template", Messages: []api.Message{{Role: "user", Content: "hello"}}, Stream: &stream},
			want:    2.5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := createRequest(t, tt.handler, tt.body)
			if w.Code != http.StatusOK {
				t.Fatalf("status = %d: %s", w.Code, w.Body.String())
			}

			var got api.Metrics
			if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
				t.Fatal(err)
			}
			if got.GPUEnergy != tt.want {
				t.Errorf("gpu_energy = %v, want %v", got.GPUEnergy, tt.want)
			}
		})
	}
}
