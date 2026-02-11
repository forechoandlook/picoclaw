package providers

import (
	"context"
	"fmt"

	"github.com/sipeed/picoclaw/pkg/config"
)

type OllamaProvider struct {
	config       config.OllamaConfig
	httpProvider *HTTPProvider
}

func NewOllamaProvider(cfg config.OllamaConfig) (*OllamaProvider, error) {
	p := &OllamaProvider{
		config: cfg,
	}

	if cfg.Mode == "http" {
		if cfg.APIBase == "" {
			return nil, fmt.Errorf("api_base is required for ollama http mode")
		}
		// Ollama usually doesn't need an API key for local usage, but we pass empty string
		p.httpProvider = NewHTTPProvider("", cfg.APIBase)
	}

	return p, nil
}

func (p *OllamaProvider) Chat(ctx context.Context, messages []Message, tools []ToolDefinition, model string, options map[string]interface{}) (*LLMResponse, error) {
	if p.config.Mode == "http" {
		return p.httpProvider.Chat(ctx, messages, tools, model, options)
	}

	if p.config.Mode == "local" {
		return nil, fmt.Errorf("local mode for ollama is not yet fully supported in this build. Please use http mode or manually link llama.cpp")

		/*
		// Local mode implementation would go here, requiring static linking of llama.cpp
		// which is complex to set up in this environment.
		// Example pseudo-code:

		if p.llm == nil {
			p.llm, err = llama.New(p.config.ModelPath, ...)
		}
		text, err := p.llm.Predict(...)
		return ...
		*/
	}

	return nil, fmt.Errorf("unknown ollama mode: %s", p.config.Mode)
}

func (p *OllamaProvider) GetDefaultModel() string {
	return "llama-2"
}
