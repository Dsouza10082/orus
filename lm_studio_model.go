package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-chi/chi/v5/middleware"
	"github.com/gorilla/websocket"
	lmstudio "github.com/liliang-cn/lmstudio-go"
)

type LMStudioTask struct {
	ID       string
	Request  LMStudioChatRequest
	Callback func(LMStudioStreamResponse)
	Ctx      context.Context
}

type LMStudioChatRequest struct {
	Model            string             `json:"model"`
	Messages         []lmstudio.Message `json:"messages"`
	Temperature      float64            `json:"temperature,omitempty"`
	MaxTokens        int                `json:"max_tokens,omitempty"`
	TopP             float64            `json:"top_p,omitempty"`
	FrequencyPenalty float64            `json:"frequency_penalty,omitempty"`
	PresencePenalty  float64            `json:"presence_penalty,omitempty"`
	Stream           bool               `json:"stream,omitempty"`
	Stop             []string           `json:"stop,omitempty"`
	N                int                `json:"n,omitempty"`
	User             string             `json:"user,omitempty"`
}

type LMStudioChatResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

type Choice struct {
	Index        int               `json:"index"`
	Message      *lmstudio.Message `json:"message,omitempty"`
	Delta        *lmstudio.Message `json:"delta,omitempty"`
	FinishReason *string           `json:"finish_reason"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type LMStudioWorkerPool struct {
	workers   int
	taskQueue chan LMStudioTask
	wg        sync.WaitGroup
	shutdown  atomic.Bool
	client    *LMStudioClient
	mu        sync.RWMutex

	tasksProcessed atomic.Int64
	tasksQueued    atomic.Int64
	tasksFailed    atomic.Int64
}

type LMStudioPullRequest struct {
	Model       string `json:"model"`        // ID ou nome do modelo (ex: "TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
	Filename    string `json:"filename,omitempty"` // Arquivo específico (ex: "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
	Quantization string `json:"quantization,omitempty"` // Quantização preferida (ex: "Q4_K_M", "Q5_K_M", "Q8_0")
}

// PullResponse resposta do pull
type LMStudioPullResponse struct {
	Success  bool            `json:"success"`
	Message  string          `json:"message"`
	ModelID  string          `json:"model_id,omitempty"`
	Progress *DownloadProgress `json:"progress,omitempty"`
}

type DownloadProgress struct {
	ModelID        string    `json:"model_id"`
	Status         string    `json:"status"` // pending, downloading, completed, failed
	Progress       float64   `json:"progress"` // 0-100
	BytesDownloaded int64    `json:"bytes_downloaded"`
	TotalBytes     int64     `json:"total_bytes"`
	Speed          string    `json:"speed,omitempty"`
	ETA            string    `json:"eta,omitempty"`
	Error          string    `json:"error,omitempty"`
	StartedAt      time.Time `json:"started_at"`
	CompletedAt    *time.Time `json:"completed_at,omitempty"`
}

type LMStudioLoadModelRequest struct {
	Model       string `json:"model"`
	ContextSize int    `json:"context_size,omitempty"`
	GPULayers   int    `json:"gpu_layers,omitempty"`
}

type AdaptiveRateLimiter struct {
	tokens     atomic.Int64
	maxTokens  int64
	refillRate int64 // tokens por segundo
	lastRefill atomic.Int64

	// Métricas para adaptação
	successCount atomic.Int64
	failureCount atomic.Int64
	avgLatency   atomic.Int64

	// Configuração adaptativa
	minRate       int64
	maxRate       int64
	adaptInterval time.Duration

	mu       sync.Mutex
	stopChan chan struct{}
	wg       sync.WaitGroup
}

type LMStudioChatHandler struct {
	client      *LMStudioClient
	rateLimiter *AdaptiveRateLimiter
	logger      *slog.Logger
	upgrader    websocket.Upgrader
}

// RateLimiterConfig configuração do rate limiter
type RateLimiterConfig struct {
	InitialTokens int64
	MaxTokens     int64
	RefillRate    int64 // tokens por segundo
	MinRate       int64
	MaxRate       int64
	AdaptInterval time.Duration
}

type HuggingFaceModel struct {
	ID           string   `json:"id"`
	ModelID      string   `json:"modelId"`
	Author       string   `json:"author,omitempty"`
	SHA          string   `json:"sha,omitempty"`
	LastModified string   `json:"lastModified,omitempty"`
	Private      bool     `json:"private"`
	Disabled     bool     `json:"disabled,omitempty"`
	Gated        bool     `json:"gated,omitempty"`
	Downloads    int      `json:"downloads"`
	Likes        int      `json:"likes"`
	Tags         []string `json:"tags,omitempty"`
	PipelineTag  string   `json:"pipeline_tag,omitempty"`
	LibraryName  string   `json:"library_name,omitempty"`
}

func DefaultRateLimiterConfig() RateLimiterConfig {
	return RateLimiterConfig{
		InitialTokens: 16,
		MaxTokens:     32,
		RefillRate:    8,
		MinRate:       2,
		MaxRate:       16,
		AdaptInterval: 5 * time.Second,
	}
}

func (h *LMStudioChatHandler) ChatCompletions(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	requestID := middleware.GetReqID(ctx)

	var req LMStudioChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		h.logger.Error("Invalid request", "error", err, "request_id", requestID)
		respondError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}

	// Se stream=true, redireciona para streaming
	if req.Stream {
		h.ChatCompletionsStream(w, r)
		return
	}

	h.logger.Info("Chat completion request",
		"request_id", requestID,
		"model", req.Model,
		"messages", len(req.Messages),
	)

	// Converte para request do cliente
	lmReq := h.toLMStudioRequest(req)

	// Executa streaming internamente e coleta resposta
	startTime := time.Now()
	streamChan, err := h.client.ChatCompletionStream(ctx, lmReq)
	if err != nil {
		h.logger.Error("Stream error", "error", err, "request_id", requestID)
		respondError(w, http.StatusInternalServerError, "stream_error", err.Error())
		return
	}

	var content string
	var tokenCount int

	for chunk := range streamChan {
		if chunk.Error != nil {
			h.logger.Error("Chunk error", "error", chunk.Error, "request_id", requestID)
			respondError(w, http.StatusInternalServerError, "chunk_error", chunk.Error.Error())
			return
		}
		if chunk.Done {
			break
		}
		if chunk.Chunk != nil && len(chunk.Chunk.Choices) > 0 {
			content += chunk.Chunk.Choices[0].Delta.Content
			tokenCount++
		}
	}

	latency := time.Since(startTime)
	h.rateLimiter.RecordSuccess(latency.Milliseconds())

	// Monta resposta
	finishReason := "stop"
	response := LMStudioChatResponse{
		ID:      fmt.Sprintf("chatcmpl-%s", requestID),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []Choice{
			{
				Index: 0,
				Message: &lmstudio.Message{
					Role:    "assistant",
					Content: content,
				},
				FinishReason: &finishReason,
			},
		},
		Usage: Usage{
			PromptTokens:     len(req.Messages) * 10, // Estimativa
			CompletionTokens: tokenCount,
			TotalTokens:      len(req.Messages)*10 + tokenCount,
		},
	}

	respondJSON(w, http.StatusOK, response)
}

func (h *LMStudioChatHandler) ChatCompletionsStream(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	requestID := middleware.GetReqID(ctx)

	var req LMStudioChatRequest

	// Tenta ler do body se não foi processado ainda
	if r.Body != nil {
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil && err != io.EOF {
			h.logger.Error("Invalid request", "error", err, "request_id", requestID)
			respondError(w, http.StatusBadRequest, "invalid_request", err.Error())
			return
		}
	}

	h.logger.Info("Chat stream request",
		"request_id", requestID,
		"model", req.Model,
		"messages", len(req.Messages),
	)

	// Configura headers SSE
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no") // Desabilita buffering no nginx
	w.Header().Set("Access-Control-Allow-Origin", "*")

	flusher, ok := w.(http.Flusher)
	if !ok {
		h.logger.Error("Streaming unsupported", "request_id", requestID)
		respondError(w, http.StatusInternalServerError, "streaming_unsupported", "Streaming not supported")
		return
	}

	// Converte para request do cliente
	lmReq := h.toLMStudioRequest(req)
	lmReq.Stream = true

	startTime := time.Now()
	streamChan, err := h.client.ChatCompletionStream(ctx, lmReq)
	if err != nil {
		h.writeSSEError(w, flusher, err)
		return
	}

	// Processa stream
	for {
		select {
		case <-ctx.Done():
			h.writeSSEData(w, flusher, "[DONE]")
			return

		case chunk, ok := <-streamChan:
			if !ok {
				h.writeSSEData(w, flusher, "[DONE]")
				return
			}

			if chunk.Error != nil {
				h.writeSSEError(w, flusher, chunk.Error)
				return
			}

			if chunk.Done {
				latency := time.Since(startTime)
				h.rateLimiter.RecordSuccess(latency.Milliseconds())
				h.writeSSEData(w, flusher, "[DONE]")
				return
			}

			if chunk.Chunk != nil {
				// Converte para formato OpenAI
				response := h.toStreamResponse(chunk.Chunk, requestID, req.Model)
				data, _ := json.Marshal(response)
				h.writeSSEData(w, flusher, string(data))
			}
		}
	}
}

func (h *LMStudioChatHandler) BatchChat(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	requestID := middleware.GetReqID(ctx)

	var requests []LMStudioChatRequest
	if err := json.NewDecoder(r.Body).Decode(&requests); err != nil {
		h.logger.Error("Invalid request", "error", err, "request_id", requestID)
		respondError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}

	if len(requests) == 0 {
		h.logger.Error("Empty batch", "request_id", requestID)
		respondError(w, http.StatusBadRequest, "empty_batch", "No requests in batch")
		return
	}

	if len(requests) > 50 {
		h.logger.Error("Batch too large", "request_id", requestID)
		respondError(w, http.StatusBadRequest, "batch_too_large", "Maximum 50 requests per batch")
		return
	}

	h.logger.Info("Batch chat request",
		"request_id", requestID,
		"batch_size", len(requests),
	)

	results := make([]LMStudioChatResponse, len(requests))
	errors := make([]error, len(requests))
	var wg sync.WaitGroup

	sem := make(chan struct{}, 8)

	for i, req := range requests {
		wg.Add(1)
		go func(idx int, chatReq LMStudioChatRequest) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			lmReq := h.toLMStudioRequest(chatReq)
			streamChan, err := h.client.ChatCompletionStream(ctx, lmReq)
			if err != nil {
				errors[idx] = err
				return
			}

			var content string
			var tokenCount int

			for chunk := range streamChan {
				if chunk.Error != nil {
					errors[idx] = chunk.Error
					return
				}
				if chunk.Done {
					break
				}
				if chunk.Chunk != nil && len(chunk.Chunk.Choices) > 0 {
					content += chunk.Chunk.Choices[0].Delta.Content
					tokenCount++
				}
			}

			finishReason := "stop"
			results[idx] = LMStudioChatResponse{
				ID:      fmt.Sprintf("chatcmpl-%s-%d", requestID, idx),
				Object:  "chat.completion",
				Created: time.Now().Unix(),
				Model:   chatReq.Model,
				Choices: []Choice{
					{
						Index: 0,
						Message: &lmstudio.Message{
							Role:    "assistant",
							Content: content,
						},
						FinishReason: &finishReason,
					},
				},
				Usage: Usage{
					CompletionTokens: tokenCount,
				},
			}
		}(i, req)
	}

	wg.Wait()

	// Monta resposta batch
	type BatchResponse struct {
		Results []LMStudioChatResponse `json:"results"`
		Errors  []string               `json:"errors,omitempty"`
	}

	batchResp := BatchResponse{
		Results: results,
	}

	for i, err := range errors {
		if err != nil {
			batchResp.Errors = append(batchResp.Errors, fmt.Sprintf("[%d] %v", i, err))
		}
	}

	respondJSON(w, http.StatusOK, batchResp)
}

func (h *LMStudioChatHandler) ListModels(w http.ResponseWriter, r *http.Request) {
	type Model struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}

	type ModelsResponse struct {
		Object string  `json:"object"`
		Data   []Model `json:"data"`
	}

	response := ModelsResponse{
		Object: "list",
		Data: []Model{
			{
				ID:      "local-model",
				Object:  "model",
				Created: time.Now().Unix(),
				OwnedBy: "local",
			},
		},
	}

	respondJSON(w, http.StatusOK, response)
}

func (h *LMStudioChatHandler) WebSocketChat(w http.ResponseWriter, r *http.Request) {
	conn, err := h.upgrader.Upgrade(w, r, nil)
	if err != nil {
		h.logger.Error("WebSocket upgrade error", "error", err)
		return
	}
	defer conn.Close()

	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	h.logger.Info("WebSocket connection established", "remote", r.RemoteAddr)

	for {
		var req LMStudioChatRequest
		if err := conn.ReadJSON(&req); err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				h.logger.Error("WebSocket read error", "error", err)
			}
			return
		}

		go h.handleWebSocketRequest(ctx, conn, req)
	}
}

func (h *LMStudioChatHandler) handleWebSocketRequest(ctx context.Context, conn *websocket.Conn, req LMStudioChatRequest) {
	lmReq := h.toLMStudioRequest(req)
	lmReq.Stream = true

	streamChan, err := h.client.ChatCompletionStream(ctx, LMStudioChatRequest{
		Model:    req.Model,
		Messages: req.Messages,
		Stream:   true,
	})
	if err != nil {
		conn.WriteJSON(map[string]string{"error": err.Error()})
		return
	}

	for chunk := range streamChan {
		if chunk.Error != nil {
			conn.WriteJSON(map[string]string{"error": chunk.Error.Error()})
			return
		}

		if chunk.Done {
			conn.WriteJSON(map[string]string{"event": "done"})
			return
		}

		if chunk.Chunk != nil {
			response := h.toStreamResponse(chunk.Chunk, "ws", req.Model)
			if err := conn.WriteJSON(response); err != nil {
				h.logger.Error("WebSocket write error", "error", err)
				return
			}
		}
	}
}

// toLMStudioRequest converte request da API para request do cliente
func (h *LMStudioChatHandler) toLMStudioRequest(req LMStudioChatRequest) LMStudioChatRequest {
	messages := make([]lmstudio.Message, len(req.Messages))
	for i, msg := range req.Messages {
		messages[i] = lmstudio.Message{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}

	model := req.Model
	if model == "" {
		model = "local-model"
	}

	return LMStudioChatRequest{
		Model:    model,
		Messages: messages,
		Stop:     req.Stop,
		Stream:   true,
	}
}

// toStreamResponse converte chunk para formato OpenAI stream
func (h *LMStudioChatHandler) toStreamResponse(chunk *LMStudioStreamChunk, requestID, model string) LMStudioChatResponse {
	var choices []Choice

	for _, c := range chunk.Choices {
		choices = append(choices, Choice{
			Index: c.Index,
			Delta: &lmstudio.Message{
				Role:    c.Delta.Role,
				Content: c.Delta.Content,
			},
			FinishReason: c.FinishReason,
		})
	}

	return LMStudioChatResponse{
		ID:      fmt.Sprintf("chatcmpl-%s", requestID),
		Object:  "chat.completion.chunk",
		Created: chunk.Created,
		Model:   model,
		Choices: choices,
	}
}

func NewLMStudioChatHandler(client *LMStudioClient, rl *AdaptiveRateLimiter, logger *slog.Logger) *LMStudioChatHandler {
	return &LMStudioChatHandler{
		client:      client,
		rateLimiter: rl,
		logger:      logger,
		upgrader: websocket.Upgrader{
			ReadBufferSize:  1024,
			WriteBufferSize: 1024,
			CheckOrigin: func(r *http.Request) bool {
				return true
			},
		},
	}
}

func NewAdaptiveRateLimiter(cfg RateLimiterConfig) *AdaptiveRateLimiter {
	rl := &AdaptiveRateLimiter{
		maxTokens:     cfg.MaxTokens,
		refillRate:    cfg.RefillRate,
		minRate:       cfg.MinRate,
		maxRate:       cfg.MaxRate,
		adaptInterval: cfg.AdaptInterval,
		stopChan:      make(chan struct{}),
	}

	rl.tokens.Store(cfg.InitialTokens)
	rl.lastRefill.Store(time.Now().UnixNano())

	return rl
}

func (rl *AdaptiveRateLimiter) Start() {
	rl.wg.Add(2)

	// Goroutine de refill
	go func() {
		defer rl.wg.Done()
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				rl.refill()
			case <-rl.stopChan:
				return
			}
		}
	}()

	// Goroutine de adaptação
	go func() {
		defer rl.wg.Done()
		ticker := time.NewTicker(rl.adaptInterval)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				rl.adapt()
			case <-rl.stopChan:
				return
			}
		}
	}()
}

func (rl *AdaptiveRateLimiter) refill() {
	current := rl.tokens.Load()
	newTokens := current + rl.refillRate
	if newTokens > rl.maxTokens {
		newTokens = rl.maxTokens
	}
	rl.tokens.Store(newTokens)
}

// adapt ajusta a taxa baseado no desempenho
func (rl *AdaptiveRateLimiter) adapt() {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	success := rl.successCount.Swap(0)
	failures := rl.failureCount.Swap(0)

	total := success + failures
	if total == 0 {
		return
	}

	failureRate := float64(failures) / float64(total)

	// Ajusta taxa baseado na taxa de falha
	if failureRate > 0.2 {
		// Muitas falhas, reduz taxa
		newRate := rl.refillRate - 1
		if newRate < rl.minRate {
			newRate = rl.minRate
		}
		rl.refillRate = newRate
	} else if failureRate < 0.05 {
		// Poucas falhas, aumenta taxa
		newRate := rl.refillRate + 1
		if newRate > rl.maxRate {
			newRate = rl.maxRate
		}
		rl.refillRate = newRate
	}
}

// Acquire tenta adquirir um token
func (rl *AdaptiveRateLimiter) Acquire(ctx context.Context) error {
	for {
		current := rl.tokens.Load()
		if current > 0 {
			if rl.tokens.CompareAndSwap(current, current-1) {
				return nil
			}
			continue
		}

		// Espera por token disponível
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(100 * time.Millisecond):
			// Tenta novamente
		}
	}
}

// TryAcquire tenta adquirir sem bloquear
func (rl *AdaptiveRateLimiter) TryAcquire() bool {
	for {
		current := rl.tokens.Load()
		if current <= 0 {
			return false
		}
		if rl.tokens.CompareAndSwap(current, current-1) {
			return true
		}
	}
}

// RecordSuccess registra uma operação bem-sucedida
func (rl *AdaptiveRateLimiter) RecordSuccess(latencyMs int64) {
	rl.successCount.Add(1)

	// Atualiza média de latência (média móvel simples)
	current := rl.avgLatency.Load()
	if current == 0 {
		rl.avgLatency.Store(latencyMs)
	} else {
		rl.avgLatency.Store((current + latencyMs) / 2)
	}
}

// RecordFailure registra uma falha
func (rl *AdaptiveRateLimiter) RecordFailure() {
	rl.failureCount.Add(1)
}

// Stats retorna estatísticas do rate limiter
func (rl *AdaptiveRateLimiter) Stats() RateLimiterStats {
	return RateLimiterStats{
		AvailableTokens: rl.tokens.Load(),
		MaxTokens:       rl.maxTokens,
		RefillRate:      rl.refillRate,
		AvgLatencyMs:    rl.avgLatency.Load(),
	}
}

// RateLimiterStats estatísticas do rate limiter
type RateLimiterStats struct {
	AvailableTokens int64
	MaxTokens       int64
	RefillRate      int64
	AvgLatencyMs    int64
}

// Stop para o rate limiter
func (rl *AdaptiveRateLimiter) Stop() {
	close(rl.stopChan)
	rl.wg.Wait()
}

// SlidingWindowCounter implementa contador de janela deslizante
type SlidingWindowCounter struct {
	windowSize time.Duration
	buckets    []atomic.Int64
	bucketSize time.Duration
	numBuckets int
	currentIdx atomic.Int64
	lastRotate atomic.Int64
	mu         sync.Mutex
}

// NewSlidingWindowCounter cria um novo contador de janela deslizante
func NewSlidingWindowCounter(windowSize time.Duration, numBuckets int) *SlidingWindowCounter {
	swc := &SlidingWindowCounter{
		windowSize: windowSize,
		numBuckets: numBuckets,
		bucketSize: windowSize / time.Duration(numBuckets),
		buckets:    make([]atomic.Int64, numBuckets),
	}
	swc.lastRotate.Store(time.Now().UnixNano())
	return swc
}

// Increment incrementa o contador
func (swc *SlidingWindowCounter) Increment() {
	swc.rotate()
	idx := swc.currentIdx.Load() % int64(swc.numBuckets)
	swc.buckets[idx].Add(1)
}

// Count retorna a contagem total na janela
func (swc *SlidingWindowCounter) Count() int64 {
	swc.rotate()
	var total int64
	for i := range swc.buckets {
		total += swc.buckets[i].Load()
	}
	return total
}

// rotate rotaciona os buckets se necessário
func (swc *SlidingWindowCounter) rotate() {
	now := time.Now().UnixNano()
	last := swc.lastRotate.Load()
	elapsed := time.Duration(now - last)

	if elapsed < swc.bucketSize {
		return
	}

	swc.mu.Lock()
	defer swc.mu.Unlock()

	// Verifica novamente após adquirir o lock
	last = swc.lastRotate.Load()
	elapsed = time.Duration(now - last)

	bucketsToRotate := int(elapsed / swc.bucketSize)
	if bucketsToRotate == 0 {
		return
	}

	if bucketsToRotate >= swc.numBuckets {
		// Reset todos os buckets
		for i := range swc.buckets {
			swc.buckets[i].Store(0)
		}
	} else {
		// Rotaciona apenas os buckets necessários
		for i := 0; i < bucketsToRotate; i++ {
			idx := (swc.currentIdx.Load() + int64(i) + 1) % int64(swc.numBuckets)
			swc.buckets[idx].Store(0)
		}
	}

	swc.currentIdx.Add(int64(bucketsToRotate))
	swc.lastRotate.Store(now)
}

func NewLMStudioWorkerPool(workers, queueSize int) *LMStudioWorkerPool {
	return &LMStudioWorkerPool{
		workers:   workers,
		taskQueue: make(chan LMStudioTask, queueSize),
	}
}

func (wp *LMStudioWorkerPool) Start(client *LMStudioClient) {
	wp.mu.Lock()
	wp.client = client
	wp.mu.Unlock()

	for i := 0; i < wp.workers; i++ {
		wp.wg.Add(1)
		go wp.worker(i)
	}
}

func (wp *LMStudioWorkerPool) worker(id int) {
	defer wp.wg.Done()

	for task := range wp.taskQueue {
		if wp.shutdown.Load() {
			return
		}

		wp.processTask(task)
	}
}

func (wp *LMStudioWorkerPool) processTask(task LMStudioTask) {
	wp.mu.RLock()
	client := wp.client
	wp.mu.RUnlock()

	if client == nil {
		wp.tasksFailed.Add(1)
		task.Callback(LMStudioStreamResponse{Error: errors.New("client not set")})
		return
	}

	ctx := task.Ctx
	if ctx == nil {
		ctx = context.Background()
	}

	streamChan, err := client.ChatCompletionStream(ctx, task.Request)
	if err != nil {
		wp.tasksFailed.Add(1)
		task.Callback(LMStudioStreamResponse{Error: err})
		return
	}

	for response := range streamChan {
		task.Callback(response)
		if response.Done || response.Error != nil {
			break
		}
	}

	wp.tasksProcessed.Add(1)
}

func (wp *LMStudioWorkerPool) Submit(task LMStudioTask) bool {
	if wp.shutdown.Load() {
		return false
	}

	select {
	case wp.taskQueue <- task:
		wp.tasksQueued.Add(1)
		return true
	default:
		return false
	}
}

func (wp *LMStudioWorkerPool) SubmitWait(ctx context.Context, task LMStudioTask) error {
	if wp.shutdown.Load() {
		return errors.New("pool shutdown")
	}

	select {
	case wp.taskQueue <- task:
		wp.tasksQueued.Add(1)
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (wp *LMStudioWorkerPool) Shutdown() {
	if wp.shutdown.CompareAndSwap(false, true) {
		close(wp.taskQueue)
		wp.wg.Wait()
	}
}

func (wp *LMStudioWorkerPool) Stats() PoolStats {
	return PoolStats{
		Workers:        wp.workers,
		QueueSize:      len(wp.taskQueue),
		QueueCapacity:  cap(wp.taskQueue),
		TasksProcessed: wp.tasksProcessed.Load(),
		TasksQueued:    wp.tasksQueued.Load(),
		TasksFailed:    wp.tasksFailed.Load(),
		IsShutdown:     wp.shutdown.Load(),
	}
}

type PoolStats struct {
	Workers        int
	QueueSize      int
	QueueCapacity  int
	TasksProcessed int64
	TasksQueued    int64
	TasksFailed    int64
	IsShutdown     bool
}

type BatchProcessor struct {
	client      *LMStudioClient
	maxParallel int
	results     chan BatchResult
}

type BatchResult struct {
	ID       string
	Response string
	Error    error
	Tokens   int
}

func NewBatchProcessor(client *LMStudioClient, maxParallel int) *BatchProcessor {
	return &BatchProcessor{
		client:      client,
		maxParallel: maxParallel,
		results:     make(chan BatchResult, maxParallel*2),
	}
}

func (bp *BatchProcessor) ProcessBatch(ctx context.Context, requests []LMStudioChatRequest) <-chan BatchResult {
	results := make(chan BatchResult, len(requests))

	go func() {
		defer close(results)

		sem := make(chan struct{}, bp.maxParallel)
		var wg sync.WaitGroup

		for i, req := range requests {
			select {
			case <-ctx.Done():
				results <- BatchResult{
					ID:    req.Model,
					Error: ctx.Err(),
				}
				return
			case sem <- struct{}{}:
			}

			wg.Add(1)
			go func(idx int, request LMStudioChatRequest) {
				defer func() {
					<-sem
					wg.Done()
				}()

				bp.processSingleRequest(ctx, idx, request, results)
			}(i, req)
		}

		wg.Wait()
	}()

	return results
}

func (bp *BatchProcessor) processSingleRequest(ctx context.Context, idx int, req LMStudioChatRequest, results chan<- BatchResult) {
	streamChan, err := bp.client.ChatCompletionStream(ctx, req)
	if err != nil {
		results <- BatchResult{
			ID:    req.Model,
			Error: err,
		}
		return
	}

	var response strings.Builder
	var tokenCount int

	for chunk := range streamChan {
		if chunk.Error != nil {
			results <- BatchResult{
				ID:    req.Model,
				Error: chunk.Error,
			}
			return
		}

		if chunk.Done {
			break
		}

		if chunk.Chunk != nil && len(chunk.Chunk.Choices) > 0 {
			content := chunk.Chunk.Choices[0].Delta.Content
			response.WriteString(content)
			if content != "" {
				tokenCount++
			}
		}
	}

	results <- BatchResult{
		ID:       req.Model,
		Response: response.String(),
		Tokens:   tokenCount,
	}
}



func (api *LMStudioChatHandler) writeSSEData(w http.ResponseWriter, flusher http.Flusher, data string) {
	fmt.Fprintf(w, "data: %s\n\n", data)
	flusher.Flush()
}

func (api *LMStudioChatHandler) writeSSEError(w http.ResponseWriter, flusher http.Flusher, err error) {
	errData, _ := json.Marshal(map[string]string{"error": err.Error()})
	fmt.Fprintf(w, "data: %s\n\n", string(errData))
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func (c *LMStudioChatHandler) PullModelHandler(w http.ResponseWriter, r *http.Request) {
	var req LMStudioPullRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}

	if req.Model == "" {
		respondError(w, http.StatusBadRequest, "missing_model", "Model name is required")
		return
	}

	c.logger.Info("Pull model request",
		"model", req.Model,
		"filename", req.Filename,
		"quantization", req.Quantization,
	)

	resp, err := c.PullModel(r.Context(), req)
	if err != nil {
		respondError(w, http.StatusInternalServerError, "pull_failed", err.Error())
		return
	}

	respondJSON(w, http.StatusAccepted, resp)
}

func (c *LMStudioChatHandler) PullModel(ctx context.Context, req LMStudioPullRequest) (*LMStudioPullResponse, error) {
	// Normaliza o nome do modelo
	modelID := normalizeModelID(req.Model)
	
	// Verifica se já está baixando
	c.client.downloadsMu.RLock()
	if progress, exists := c.client.downloads[modelID]; exists {
		c.client.downloadsMu.RUnlock()
		return &LMStudioPullResponse{
			Success:  true,
			Message:  "Download already in progress",
			ModelID:  modelID,
			Progress: progress,
		}, nil
	}
	c.client.downloadsMu.RUnlock()

	// Inicia tracking do download
	progress := &DownloadProgress{
		ModelID:   modelID,
		Status:    "pending",
		Progress:  0,
		StartedAt: time.Now(),
	}

	c.client.downloadsMu.Lock()
	c.client.downloads[modelID] = progress
	c.client.downloadsMu.Unlock()

	// Tenta diferentes métodos de pull
	
	// Método 1: API interna do LM Studio (se disponível)
	if err := c.pullViaLMStudioAPI(ctx, req, progress); err == nil {
		return &LMStudioPullResponse{
			Success:  true,
			Message:  "Download started via LM Studio API",
			ModelID:  modelID,
			Progress: progress,
		}, nil
	}

	go c.pullFromHuggingFace(context.Background(), req, progress)

	return &LMStudioPullResponse{
		Success:  true,
		Message:  "Download started via Hugging Face",
		ModelID:  modelID,
		Progress: progress,
	}, nil
}

func (c *LMStudioChatHandler) pullViaLMStudioAPI(ctx context.Context, req LMStudioPullRequest, progress *DownloadProgress) error {
	// LM Studio pode ter endpoint para download de modelos
	// Endpoint típico: POST /api/v0/models/download
	
	body, err := json.Marshal(map[string]interface{}{
		"model":        req.Model,
		"filename":     req.Filename,
		"quantization": req.Quantization,
	})
	if err != nil {
		return err
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", 
		c.client.baseURL+"/api/v0/models/download", bytes.NewReader(body))
	if err != nil {
		return err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.client.httpClient.Do(httpReq)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusAccepted {
		return fmt.Errorf("LM Studio API not available for downloads")
	}

	progress.Status = "downloading"
	return nil
}

func (mm *LMStudioChatHandler) pullFromHuggingFace(ctx context.Context, req LMStudioPullRequest, progress *DownloadProgress) {
	progress.Status = "downloading"
	
	// Constrói URL do Hugging Face
	hfURL := buildHuggingFaceURL(req)
	
	// Cria request com suporte a progresso
	httpReq, err := http.NewRequestWithContext(ctx, "GET", hfURL, nil)
	if err != nil {
		progress.Status = "failed"
		progress.Error = err.Error()
		return
	}

	// Cliente com timeout maior para downloads
	client := &http.Client{Timeout: 0} // Sem timeout para downloads grandes
	resp, err := client.Do(httpReq)
	if err != nil {
		progress.Status = "failed"
		progress.Error = err.Error()
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		progress.Status = "failed"
		progress.Error = fmt.Sprintf("HTTP %d", resp.StatusCode)
		return
	}

	progress.TotalBytes = resp.ContentLength

	// Lê com tracking de progresso
	// Em produção, salvaria no diretório de modelos do LM Studio
	var downloaded int64
	buf := make([]byte, 32*1024) // 32KB buffer
	
	startTime := time.Now()
	
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			downloaded += int64(n)
			progress.BytesDownloaded = downloaded
			
			if progress.TotalBytes > 0 {
				progress.Progress = float64(downloaded) / float64(progress.TotalBytes) * 100
			}
			
			// Calcula velocidade
			elapsed := time.Since(startTime).Seconds()
			if elapsed > 0 {
				speed := float64(downloaded) / elapsed
				progress.Speed = formatBytes(int64(speed)) + "/s"
				
				if progress.TotalBytes > 0 && speed > 0 {
					remaining := float64(progress.TotalBytes-downloaded) / speed
					progress.ETA = formatDuration(time.Duration(remaining) * time.Second)
				}
			}
		}
		
		if err == io.EOF {
			break
		}
		if err != nil {
			progress.Status = "failed"
			progress.Error = err.Error()
			return
		}
	}

	now := time.Now()
	progress.Status = "completed"
	progress.Progress = 100
	progress.CompletedAt = &now
}

// GetDownloadProgress retorna o progresso de um download
func (h *LMStudioChatHandler) GetDownloadProgress(modelID string) (*DownloadProgress, bool) {
	h.client.downloadsMu.RLock()
	defer h.client.downloadsMu.RUnlock()
	
	progress, exists := h.client.downloads[normalizeModelID(modelID)]
	return progress, exists
}

// GetAllDownloads retorna todos os downloads
func (h *LMStudioChatHandler) GetAllDownloads() map[string]*DownloadProgress {
	h.client.downloadsMu.RLock()
	defer h.client.downloadsMu.RUnlock()
	
	result := make(map[string]*DownloadProgress)
	for k, v := range h.client.downloads {
		result[k] = v
	}
	return result
}

// CancelDownload cancela um download em progresso
func (h *LMStudioChatHandler) CancelDownload(modelID string) bool {
	h.client.downloadsMu.Lock()
	defer h.client.downloadsMu.Unlock()
	
	normalizedID := normalizeModelID(modelID)
	if progress, exists := h.client.downloads[normalizedID]; exists {
		if progress.Status == "downloading" || progress.Status == "pending" {
			progress.Status = "cancelled"
			return true
		}
	}
	return false
}

// ClearCompletedDownloads limpa downloads concluídos do cache
func (h *LMStudioChatHandler) ClearCompletedDownloads() int {
	h.client.downloadsMu.Lock()
	defer h.client.downloadsMu.Unlock()
	
	count := 0
	for id, progress := range h.client.downloads {
		if progress.Status == "completed" || progress.Status == "failed" || progress.Status == "cancelled" {
			delete(h.client.downloads, id)
			count++
		}
	}
	return count
}


func (c *LMStudioChatHandler) ModelLoadHandler(w http.ResponseWriter, r *http.Request) {
	var req LMStudioLoadModelRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}

	if req.Model == "" {
		respondError(w, http.StatusBadRequest, "missing_model", "Model name is required")
		return
	}

	c.logger.Info("Load model request",
		"model", req.Model,
		"context_size", req.ContextSize,
		"gpu_layers", req.GPULayers,
	)

	if err := c.LoadModel(r.Context(), req); err != nil {
		respondError(w, http.StatusInternalServerError, "load_failed", err.Error())
		return
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"success": true,
		"message": "Model loaded successfully",
		"model":   req.Model,
	})
}

func (c *LMStudioChatHandler) LoadModel(ctx context.Context, req LMStudioLoadModelRequest) error {
	body, err := json.Marshal(map[string]interface{}{
		"model":        req.Model,
		"context_size": req.ContextSize,
		"gpu_layers":   req.GPULayers,
	})
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", 
		c.client.baseURL+"/api/v0/models/load", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.client.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("failed to load model: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("failed to load model (status %d): %s", resp.StatusCode, string(respBody))
	}

	return nil
}

func (c *LMStudioChatHandler) ModelUnloadHandler(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model string `json:"model"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}

	if req.Model == "" {
		respondError(w, http.StatusBadRequest, "missing_model", "Model name is required")
		return
	}

	c.logger.Info("Unload model request", "model", req.Model)

	if err := c.UnloadModel(r.Context(), req.Model); err != nil {
		respondError(w, http.StatusInternalServerError, "unload_failed", err.Error())
		return
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"success": true,
		"message": "Model unloaded successfully",
		"model":   req.Model,
	})
}

func (c *LMStudioChatHandler) UnloadModel(ctx context.Context, modelID string) error {
	body, err := json.Marshal(map[string]string{"model": modelID})
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", 
		c.client.baseURL+"/api/v0/models/unload", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.client.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("failed to unload model: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("failed to unload model (status %d): %s", resp.StatusCode, string(respBody))
	}

	return nil
}

func (c *LMStudioChatHandler) SearchModelsHandler(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query().Get("q")
	if query == "" {
		query = r.URL.Query().Get("query")
	}
	if query == "" {
		respondError(w, http.StatusBadRequest, "missing_query", "Query parameter 'q' is required")
		return
	}

	// Limit opcional
	limit := 20
	if l := r.URL.Query().Get("limit"); l != "" {
		fmt.Sscanf(l, "%d", &limit)
		if limit <= 0 || limit > 100 {
			limit = 20
		}
	}

	models, err := c.SearchModels(r.Context(), query, limit)
	if err != nil {
		respondError(w, http.StatusInternalServerError, "search_failed", err.Error())
		return
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"object": "list",
		"query":  query,
		"count":  len(models),
		"data":   models,
	})
}


func (h *LMStudioChatHandler) SearchModels(ctx context.Context, query string, limit int) ([]HuggingFaceModel, error) {
	if limit <= 0 {
		limit = 20
	}

	// API do Hugging Face
	apiURL := fmt.Sprintf(
		"https://huggingface.co/api/models?search=%s&filter=gguf&sort=downloads&direction=-1&limit=%d",
		url.QueryEscape(query),
		limit,
	)

	req, err := http.NewRequestWithContext(ctx, "GET", apiURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := h.client.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to search models: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Hugging Face API error: %d", resp.StatusCode)
	}

	var models []HuggingFaceModel
	if err := json.NewDecoder(resp.Body).Decode(&models); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return models, nil
}

func (h *LMStudioChatHandler) GetModelFiles(ctx context.Context, modelID string) ([]HuggingFaceFile, error) {
	apiURL := fmt.Sprintf("https://huggingface.co/api/models/%s", url.PathEscape(modelID))

	req, err := http.NewRequestWithContext(ctx, "GET", apiURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := h.client.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to get model info: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("model not found: %s", modelID)
	}

	var modelInfo struct {
		Siblings []HuggingFaceFile `json:"siblings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&modelInfo); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// Filtra apenas arquivos GGUF
	var ggufFiles []HuggingFaceFile
	for _, f := range modelInfo.Siblings {
		if strings.HasSuffix(strings.ToLower(f.Filename), ".gguf") {
			ggufFiles = append(ggufFiles, f)
		}
	}

	return ggufFiles, nil
}

// HuggingFaceFile arquivo no Hugging Face
type HuggingFaceFile struct {
	Filename string `json:"rfilename"`
	Size     int64  `json:"size,omitempty"`
	BlobID   string `json:"blobId,omitempty"`
	LFS      *struct {
		Size        int64  `json:"size"`
		SHA256      string `json:"sha256"`
		PointerSize int    `json:"pointerSize"`
	} `json:"lfs,omitempty"`
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

func buildHuggingFaceURL(req LMStudioPullRequest) string {
	model := normalizeModelID(req.Model)
	filename := req.Filename
	
	// Se não especificou arquivo, tenta inferir
	if filename == "" {
		// Usa quantização padrão Q4_K_M se não especificada
		quant := req.Quantization
		if quant == "" {
			quant = "Q4_K_M"
		}
		// Extrai nome base do modelo
		parts := strings.Split(model, "/")
		baseName := parts[len(parts)-1]
		baseName = strings.TrimSuffix(baseName, "-GGUF")
		baseName = strings.ToLower(baseName)
		filename = fmt.Sprintf("%s.%s.gguf", baseName, quant)
	}
	
	return fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", model, filename)
}

func formatBytes(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}

func formatDuration(d time.Duration) string {
	if d < time.Minute {
		return fmt.Sprintf("%ds", int(d.Seconds()))
	}
	if d < time.Hour {
		return fmt.Sprintf("%dm %ds", int(d.Minutes()), int(d.Seconds())%60)
	}
	return fmt.Sprintf("%dh %dm", int(d.Hours()), int(d.Minutes())%60)
}