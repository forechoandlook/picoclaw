package providers

import (
	"context"
	"fmt"
	"path/filepath"
	"sync"

	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/sipeed/picoclaw/pkg/config"
)

type YzmaProvider struct {
	config config.YzmaConfig
	model  llama.Model
	vocab  llama.Vocab
	mu     sync.Mutex
}

func NewYzmaProvider(cfg config.YzmaConfig) (*YzmaProvider, error) {
	if cfg.LibPath != "" {
		if err := llama.Load(cfg.LibPath); err != nil {
			return nil, fmt.Errorf("unable to load library from %s: %w", cfg.LibPath, err)
		}
	} else {
		// Try to load with default lookup
		// Ignore error as it might be already loaded or linked
		_ = llama.Load("")
	}

	// Initialize llama backend
	llama.LogSet(llama.LogSilent())
	llama.Init()

	mParams := llama.ModelDefaultParams()
	if cfg.GpuLayers > 0 {
		mParams.NGpuLayers = int32(cfg.GpuLayers)
	} else if cfg.GpuLayers == -1 {
		// -1 usually means all layers in some bindings, but let's check llama.cpp
		// set to a high number to offload all
		mParams.NGpuLayers = 999
	}

	model, err := llama.ModelLoadFromFile(cfg.ModelPath, mParams)
	if err != nil {
		return nil, fmt.Errorf("unable to load model from %s: %w", cfg.ModelPath, err)
	}

	vocab := llama.ModelGetVocab(model)

	return &YzmaProvider{
		config: cfg,
		model:  model,
		vocab:  vocab,
	}, nil
}

func (p *YzmaProvider) Chat(ctx context.Context, messages []Message, tools []ToolDefinition, model string, options map[string]interface{}) (*LLMResponse, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Initialize context for this chat session
	ctxParams := llama.ContextDefaultParams()
	if p.config.ContextSize > 0 {
		ctxParams.NCtx = uint32(p.config.ContextSize)
	} else {
		ctxParams.NCtx = 4096 // Default
	}
	// Set batch size for context processing
	ctxParams.NBatch = 512

	lctx, err := llama.InitFromModel(p.model, ctxParams)
	if err != nil {
		return nil, fmt.Errorf("unable to initialize context: %w", err)
	}
	defer llama.Free(lctx)

	// Initialize sampler with default chain
	sp := llama.DefaultSamplerParams()
	samplers := []llama.SamplerType{
		llama.SamplerTypeTopK,
		llama.SamplerTypeTopP,
		llama.SamplerTypeMinP,
		llama.SamplerTypeTemperature,
	}
	sampler := llama.NewSampler(p.model, samplers, sp)

	// Convert messages
	chatMsgs := make([]llama.ChatMessage, 0, len(messages))
	for _, m := range messages {
		chatMsgs = append(chatMsgs, llama.NewChatMessage(m.Role, m.Content))
	}

	// Determine template
	tmpl := llama.ModelChatTemplate(p.model, "")
	if tmpl == "" {
		tmpl = "chatml" // fallback
	}

	// Apply template
	bufSize := p.config.ContextSize * 4
	if bufSize == 0 {
		bufSize = 16384
	}
	buf := make([]byte, bufSize)
	reqLen := llama.ChatApplyTemplate(tmpl, chatMsgs, true, buf)

	if reqLen > int32(len(buf)) {
		buf = make([]byte, reqLen)
		reqLen = llama.ChatApplyTemplate(tmpl, chatMsgs, true, buf)
	}

	if reqLen < 0 {
		return nil, fmt.Errorf("failed to apply chat template")
	}

	prompt := string(buf[:reqLen])

	// Tokenize
	tokens := llama.Tokenize(p.vocab, prompt, true, true)

	// Evaluate prompt in batches
	nBatch := int32(512)
	batch := llama.BatchInit(nBatch, 0, 1)
	defer llama.BatchFree(batch)

	for i := 0; i < len(tokens); i += int(nBatch) {
		batch.Clear()

		end := i + int(nBatch)
		if end > len(tokens) {
			end = len(tokens)
		}

		chunkLen := int32(end - i)

		for j := 0; j < int(chunkLen); j++ {
			pos := int32(i + j)
			logits := false
			// We need logits for the last token of the prompt to start generation
			if i+j == len(tokens)-1 {
				logits = true
			}

			batch.Add(tokens[i+j], llama.Pos(pos), []llama.SeqId{0}, logits)
		}

		ret, err := llama.Decode(lctx, batch)
		if err != nil {
			return nil, fmt.Errorf("llama_decode error: %w", err)
		}
		if ret != 0 {
			return nil, fmt.Errorf("llama_decode failed with %d", ret)
		}
	}

	// Generation
	maxTokens := 1024
	if mt, ok := options["max_tokens"].(int); ok {
		maxTokens = mt
	}

	var generatedContent string
	currPos := int32(len(tokens))

	for i := 0; i < maxTokens; i++ {
		// Sample next token
		token := llama.SamplerSample(sampler, lctx, -1)

		if llama.VocabIsEOG(p.vocab, token) {
			break
		}

		// Decode token to string
		pieceBuf := make([]byte, 256)
		l := llama.TokenToPiece(p.vocab, token, pieceBuf, 0, false)
		text := string(pieceBuf[:l])
		generatedContent += text

		// Prepare next batch with single token
		batch.Clear()
		batch.Add(token, llama.Pos(currPos), []llama.SeqId{0}, true)
		currPos++

		ret, err := llama.Decode(lctx, batch)
		if err != nil {
			return nil, fmt.Errorf("llama_decode error during generation: %w", err)
		}
		if ret != 0 {
			return nil, fmt.Errorf("llama_decode failed during generation with %d", ret)
		}

		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
	}

	return &LLMResponse{
		Content: generatedContent,
		Usage: &UsageInfo{
			PromptTokens:     len(tokens),
			CompletionTokens: int(currPos) - len(tokens),
			TotalTokens:      int(currPos),
		},
		FinishReason: "stop",
	}, nil
}

func (p *YzmaProvider) GetDefaultModel() string {
	return filepath.Base(p.config.ModelPath)
}
