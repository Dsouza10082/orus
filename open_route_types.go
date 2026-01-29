package main

import (
	"sync"
	"time"
)

type OpenRouteMessagesDTO struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type OpenRouteChatRequest struct {
	Model       string    `json:"model"`
	Models      []string  `json:"models,omitempty"`
	Messages    []Message `json:"messages"`
	Stream      bool      `json:"stream"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
	Temperature float64   `json:"temperature,omitempty"`
}

type OpenRoute struct {
	Model       string                 `json:"model"`
	Models      []string               `json:"models,omitempty"`
	Messages    []OpenRouteMessagesDTO `json:"messages"`
	Stream      bool                   `json:"stream"`
	MaxTokens   int                    `json:"max_tokens,omitempty"`
	Temperature float64                `json:"temperature,omitempty"`
}

type OpenRouteConfig struct {
	APIKey          string
	OpenRouteAPIKey string
	MaxConnections  int64
	RequestTimeout  time.Duration
	EnableAuth      bool
}

type StreamDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type StreamChoice struct {
	Index        int         `json:"index"`
	Delta        StreamDelta `json:"delta"`
	FinishReason *string     `json:"finish_reason,omitempty"`
}

type StreamError struct {
	Code    interface{} `json:"code,omitempty"`
	Message string      `json:"message,omitempty"`
}

type UsageDTO struct {
	PromptTokens     int `json:"prompt_tokens,omitempty"`
	CompletionTokens int `json:"completion_tokens,omitempty"`
	TotalTokens      int `json:"total_tokens,omitempty"`
}

type StreamChunk struct {
	ID       string         `json:"id,omitempty"`
	Object   string         `json:"object,omitempty"`
	Created  int64          `json:"created,omitempty"`
	Model    string         `json:"model,omitempty"`
	Provider string         `json:"provider,omitempty"`
	Choices  []StreamChoice `json:"choices,omitempty"`
	Error    *StreamError   `json:"error,omitempty"`
	Usage    *UsageDTO      `json:"usage,omitempty"`
}

type StreamEvent struct {
	Chunk *StreamChunk
	Error error
	Done  bool
}

type ConnectionStats struct {
	mu              sync.RWMutex
	ActiveStreams   int64
	TotalRequests   int64
	TotalErrors     int64
	TotalTokensSent int64
}

func NewConnectionStats() *ConnectionStats {
	return &ConnectionStats{}
}

func (cs *ConnectionStats) IncrementStreams() {
	cs.mu.Lock()
	cs.ActiveStreams++
	cs.TotalRequests++
	cs.mu.Unlock()
}

func (cs *ConnectionStats) DecrementStreams() {
	cs.mu.Lock()
	cs.ActiveStreams--
	cs.mu.Unlock()
}

func (cs *ConnectionStats) IncrementErrors() {
	cs.mu.Lock()
	cs.TotalErrors++
	cs.mu.Unlock()
}

func (cs *ConnectionStats) AddTokens(count int64) {
	cs.mu.Lock()
	cs.TotalTokensSent += count
	cs.mu.Unlock()
}

func (cs *ConnectionStats) GetStats() map[string]int64 {
	cs.mu.RLock()
	defer cs.mu.RUnlock()
	return map[string]int64{
		"active_streams":    cs.ActiveStreams,
		"total_requests":    cs.TotalRequests,
		"total_errors":      cs.TotalErrors,
		"total_tokens_sent": cs.TotalTokensSent,
	}
}

type BufferPool struct {
	pool sync.Pool
}

func NewBufferPool(size int) *BufferPool {
	return &BufferPool{
		pool: sync.Pool{
			New: func() interface{} {
				return make([]byte, 0, size)
			},
		},
	}
}

func (bp *BufferPool) Get() []byte {
	return bp.pool.Get().([]byte)[:0]
}

func (bp *BufferPool) Put(buf []byte) {
	bp.pool.Put(buf)
}

type RateLimiter struct {
	mu         sync.Mutex
	tokens     float64
	maxTokens  float64
	refillRate float64
	lastRefill time.Time
}

func NewRateLimiter(maxTokens, refillPerSecond float64) *RateLimiter {
	return &RateLimiter{
		tokens:     maxTokens,
		maxTokens:  maxTokens,
		refillRate: refillPerSecond,
		lastRefill: time.Now(),
	}
}

func (rl *RateLimiter) Allow() bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	elapsed := now.Sub(rl.lastRefill).Seconds()
	rl.tokens += elapsed * rl.refillRate
	if rl.tokens > rl.maxTokens {
		rl.tokens = rl.maxTokens
	}
	rl.lastRefill = now

	if rl.tokens >= 1 {
		rl.tokens--
		return true
	}
	return false
}
