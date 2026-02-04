package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
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