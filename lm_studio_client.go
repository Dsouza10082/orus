package main



import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type LMStudioClient struct {
	baseURL    string
	httpClient *http.Client
	semaphore  chan struct{}
	metrics    *LMStudioMetrics
	pool       *LMStudioWorkerPool
	mu         sync.RWMutex
	downloads   map[string]*DownloadProgress
	downloadsMu sync.RWMutex
}

type LMStudioMetrics struct {
	TotalRequests     atomic.Int64
	ActiveRequests    atomic.Int64
	FailedRequests    atomic.Int64
	TotalTokens       atomic.Int64
	AvgLatencyMs      atomic.Int64
	latencySum        atomic.Int64
	latencyCount      atomic.Int64
}

type LMStudioConfig struct {
	BaseURL           string
	MaxConcurrency    int
	RequestTimeout    time.Duration
	MaxIdleConns      int
	IdleConnTimeout   time.Duration
	WorkerPoolSize    int
	WorkerQueueSize   int
}

func LMStudioDefaultConfig() LMStudioConfig {
	return LMStudioConfig{
		BaseURL:           "http://localhost:1234",
		MaxConcurrency:    16, // 16 threads do 5800x
		RequestTimeout:    5 * time.Minute,
		MaxIdleConns:      32,
		IdleConnTimeout:   90 * time.Second,
		WorkerPoolSize:    16,
		WorkerQueueSize:   256,
	}
}

func NewLMStudioClient(cfg LMStudioConfig) *LMStudioClient {
	transport := &http.Transport{
		MaxIdleConns:        cfg.MaxIdleConns,
		MaxIdleConnsPerHost: cfg.MaxIdleConns,
		IdleConnTimeout:     cfg.IdleConnTimeout,
		DisableCompression:  true,
		ForceAttemptHTTP2:   false,
	}

	c := &LMStudioClient{
		baseURL: cfg.BaseURL,
		httpClient: &http.Client{
			Transport: transport,
			Timeout:   0,
		},
		semaphore: make(chan struct{}, cfg.MaxConcurrency),
		metrics:   &LMStudioMetrics{},
		downloads: make(map[string]*DownloadProgress),
	}

	c.pool = NewLMStudioWorkerPool(cfg.WorkerPoolSize, cfg.WorkerQueueSize)
	
	return c
}

type LMStudioChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type LMStudioStreamChunk struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index int `json:"index"`
		Delta struct {
			Role    string `json:"role,omitempty"`
			Content string `json:"content,omitempty"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
}

type LMStudioStreamResponse struct {
	Chunk *LMStudioStreamChunk
	Error error
	Done  bool
}

func (c *LMStudioClient) ChatCompletionStream(ctx context.Context, req LMStudioChatRequest) (<-chan LMStudioStreamResponse, error) {
	req.Stream = true
	
	select {
	case c.semaphore <- struct{}{}:
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	c.metrics.TotalRequests.Add(1)
	c.metrics.ActiveRequests.Add(1)

	responseChan := make(chan LMStudioStreamResponse, 64)

	go func() {
		defer func() {
			<-c.semaphore
			c.metrics.ActiveRequests.Add(-1)
			close(responseChan)
		}()

		startTime := time.Now()
		
		body, err := json.Marshal(req)
		if err != nil {
			c.metrics.FailedRequests.Add(1)
			responseChan <- LMStudioStreamResponse{Error: fmt.Errorf("marshal error: %w", err)}
			return
		}

		httpReq, err := http.NewRequestWithContext(ctx, "POST", 
			c.baseURL+"/v1/chat/completions", bytes.NewReader(body))
		if err != nil {
			c.metrics.FailedRequests.Add(1)
			responseChan <- LMStudioStreamResponse{Error: fmt.Errorf("request creation error: %w", err)}
			return
		}

		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Accept", "text/event-stream")
		httpReq.Header.Set("Cache-Control", "no-cache")
		httpReq.Header.Set("Connection", "keep-alive")

		resp, err := c.httpClient.Do(httpReq)
		if err != nil {
			c.metrics.FailedRequests.Add(1)
			responseChan <- LMStudioStreamResponse{Error: fmt.Errorf("request error: %w", err)}
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			c.metrics.FailedRequests.Add(1)
			bodyBytes, _ := io.ReadAll(resp.Body)
			responseChan <- LMStudioStreamResponse{
				Error: fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(bodyBytes)),
			}
			return
		}

		c.processStream(ctx, resp.Body, responseChan, startTime)
	}()

	return responseChan, nil
}

func (c *LMStudioClient) processStream(ctx context.Context, body io.Reader, responseChan chan<- LMStudioStreamResponse, startTime time.Time) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024) // Buffer maior para chunks grandes

	var tokenCount int64

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			responseChan <- LMStudioStreamResponse{Error: ctx.Err()}
			return
		default:
		}

		line := scanner.Text()
		
		if line == "" {
			continue
		}

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		
		if data == "[DONE]" {
			latency := time.Since(startTime).Milliseconds()
			c.metrics.latencySum.Add(latency)
			c.metrics.latencyCount.Add(1)
			c.metrics.AvgLatencyMs.Store(c.metrics.latencySum.Load() / c.metrics.latencyCount.Load())
			c.metrics.TotalTokens.Add(tokenCount)
			
			responseChan <- LMStudioStreamResponse{Done: true}
			return
		}

		var chunk LMStudioStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			responseChan <- LMStudioStreamResponse{Error: fmt.Errorf("unmarshal error: %w", err)}
			continue
		}

		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
			tokenCount++
		}

		responseChan <- LMStudioStreamResponse{Chunk: &chunk}
	}

	if err := scanner.Err(); err != nil {
		c.metrics.FailedRequests.Add(1)
		responseChan <- LMStudioStreamResponse{Error: fmt.Errorf("scanner error: %w", err)}
	}
}

func (c *LMStudioClient) GetMetrics() map[string]int64 {
	return map[string]int64{
		"total_requests":   c.metrics.TotalRequests.Load(),
		"active_requests":  c.metrics.ActiveRequests.Load(),
		"failed_requests":  c.metrics.FailedRequests.Load(),
		"total_tokens":     c.metrics.TotalTokens.Load(),
		"avg_latency_ms":   c.metrics.AvgLatencyMs.Load(),
	}
}

func (c *LMStudioClient) Close() error {
	c.pool.Shutdown()
	c.httpClient.CloseIdleConnections()
	return nil
}


func normalizeModelID(model string) string {
	// Remove prefixos comuns
	model = strings.TrimPrefix(model, "hf://")
	model = strings.TrimPrefix(model, "huggingface://")
	model = strings.TrimPrefix(model, "https://huggingface.co/")
	return model
}

