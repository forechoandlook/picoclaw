package providers

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"
	"sync"

	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/sipeed/picoclaw/pkg/config"
)

type activeModel struct {
	name        string
	model       llama.Model
	vocab       llama.Vocab
	contextSize int
}

type YzmaProvider struct {
	config      config.YzmaConfig
	active      *activeModel
	mu          sync.Mutex
	defaultName string
}

func NewYzmaProvider(cfg config.YzmaConfig) (*YzmaProvider, error) {
	if cfg.LibPath != "" {
		if err := llama.Load(cfg.LibPath); err != nil {
			return nil, fmt.Errorf("unable to load library from %s: %w", cfg.LibPath, err)
		}
	} else {
		// Try to load with default lookup
		_ = llama.Load("")
	}

	// Initialize llama backend
	llama.LogSet(llama.LogSilent())
	llama.Init()

	defaultName := "default"
	if cfg.ModelPath != "" {
		defaultName = filepath.Base(cfg.ModelPath)
	} else if len(cfg.Models) > 0 {
		for name := range cfg.Models {
			defaultName = name
			break
		}
	}

	return &YzmaProvider{
		config:      cfg,
		defaultName: defaultName,
	}, nil
}

func (p *YzmaProvider) loadModel(name string) (*activeModel, error) {
	// Resolve model path and config
	var modelPath string
	var contextSize int
	var gpuLayers int

	// Normalize name
	cleanName := strings.TrimPrefix(name, "yzma/")

	if mConfig, ok := p.config.Models[cleanName]; ok {
		modelPath = mConfig.Path
		contextSize = mConfig.ContextSize
		gpuLayers = mConfig.GpuLayers
	} else if cleanName == "default" || cleanName == "" || (p.config.ModelPath != "" && cleanName == filepath.Base(p.config.ModelPath)) {
		// Fallback to default config
		modelPath = p.config.ModelPath
		contextSize = p.config.ContextSize
		gpuLayers = p.config.GpuLayers
	} else {
		return nil, fmt.Errorf("model not found: %s", name)
	}

	if modelPath == "" {
		return nil, fmt.Errorf("model path not configured for %s", name)
	}

	if contextSize == 0 {
		contextSize = p.config.ContextSize
		if contextSize == 0 {
			contextSize = 4096
		}
	}
	if gpuLayers == 0 {
		gpuLayers = p.config.GpuLayers
		if gpuLayers == 0 {
			gpuLayers = -1
		}
	}

	mParams := llama.ModelDefaultParams()
	if gpuLayers > 0 {
		mParams.NGpuLayers = int32(gpuLayers)
	} else if gpuLayers == -1 {
		mParams.NGpuLayers = 999
	}

	model, err := llama.ModelLoadFromFile(modelPath, mParams)
	if err != nil {
		return nil, fmt.Errorf("unable to load model from %s: %w", modelPath, err)
	}

	vocab := llama.ModelGetVocab(model)

	return &activeModel{
		name:        name,
		model:       model,
		vocab:       vocab,
		contextSize: contextSize,
	}, nil
}

func (p *YzmaProvider) unloadActive() {
	if p.active != nil {
		if p.active.model != 0 {
			llama.ModelFree(p.active.model)
		}
		p.active = nil
	}
}

func (p *YzmaProvider) Chat(ctx context.Context, messages []Message, tools []ToolDefinition, modelName string, options map[string]interface{}) (*LLMResponse, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	targetModel := modelName
	if targetModel == "" {
		targetModel = p.defaultName
	}

	if strings.HasPrefix(targetModel, "yzma/") {
		targetModel = strings.TrimPrefix(targetModel, "yzma/")
	}

	// Switch model if needed
	if p.active == nil || (p.active.name != targetModel && targetModel != "default") {
		// Unload current
		p.unloadActive()

		// Load new
		newModel, err := p.loadModel(targetModel)
		if err != nil {
			return nil, err
		}
		p.active = newModel
	}

	am := p.active

	// Initialize context
	ctxParams := llama.ContextDefaultParams()
	ctxParams.NCtx = uint32(am.contextSize)
	ctxParams.NBatch = 512

	lctx, err := llama.InitFromModel(am.model, ctxParams)
	if err != nil {
		return nil, fmt.Errorf("unable to initialize context: %w", err)
	}
	defer llama.Free(lctx)

	// Initialize sampler
	sp := llama.DefaultSamplerParams()
	samplers := []llama.SamplerType{
		llama.SamplerTypeTopK,
		llama.SamplerTypeTopP,
		llama.SamplerTypeMinP,
		llama.SamplerTypeTemperature,
	}
	sampler := llama.NewSampler(am.model, samplers, sp)

	// Convert messages
	chatMsgs := make([]llama.ChatMessage, 0, len(messages))
	for _, m := range messages {
		chatMsgs = append(chatMsgs, llama.NewChatMessage(m.Role, m.Content))
	}

	// Template
	tmpl := llama.ModelChatTemplate(am.model, "")
	if tmpl == "" {
		tmpl = "chatml"
	}

	// Apply template
	bufSize := am.contextSize * 4
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
	tokens := llama.Tokenize(am.vocab, prompt, true, true)

	// Decode Loop
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
		token := llama.SamplerSample(sampler, lctx, -1)

		if llama.VocabIsEOG(am.vocab, token) {
			break
		}

		pieceBuf := make([]byte, 256)
		l := llama.TokenToPiece(am.vocab, token, pieceBuf, 0, false)
		text := string(pieceBuf[:l])
		generatedContent += text

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
	return p.defaultName
}
