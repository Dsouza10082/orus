package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"regexp"
	"strings"
	"sync"
	"time"
)

var OpenRouteApiUrl = LoadEnv("OPEN_ROUTE_API_URL")
var OpenRouteApiKey = LoadEnv("OPEN_ROUTE_API_KEY")

type OpenRouteWrapper struct {
	httpClient *http.Client
}

func NewOpenRouteWrapper() *OpenRouteWrapper {
	return &OpenRouteWrapper{
		httpClient: &http.Client{
			Timeout: 120 * time.Second,
		},
	}
}

func DefaultConfig() *ClientConfig {
	return &ClientConfig{
		APIKey:          OpenRouteApiKey,
		BaseURL:         OpenRouteApiUrl,
		Timeout:         120 * time.Second,
		MaxIdleConns:    100,
		MaxConnsPerHost: 100,
	}
}

type ClientConfig struct {
	APIKey         string
	BaseURL        string
	Timeout        time.Duration
	MaxIdleConns   int
	MaxConnsPerHost int
}

type Client struct {
	config     *ClientConfig
	httpClient *http.Client
	bufferPool *BufferPool
	stats      *ConnectionStats
	mu         sync.RWMutex
}

type OpenRouteResponseDTO struct {
	Id      string              `json:"id"`
	Choices []ResponseChoiceDTO `json:"choices"`
}

type ResponseChoiceDTO struct {
	Message      MessageChoiceResponseDTO `json:"message"`
	Index        int                      `json:"index"`
	FinishReason string                   `json:"finish_reason"`
	Logprobs     string                   `json:"logprobs"`
	Assistant    []*ResultService      `json:"assistant"`
}

type MessageChoiceResponseDTO struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
	Refusal string `json:"refusal"`
}

type ResultService struct {
	SerialTraining      string         `json:"serial_training" db:"serial_training"`
	SerialTrainingChild string         `json:"serial_training_child" db:"serial_training_child"`
	Answer              string         `json:"answer" db:"answer"`
	EmbeddingModel      string         `json:"embedding_model" db:"embedding_model"`
	Question            string         `json:"question" db:"question"`
	RawQuestion         string         `json:"raw_question" db:"raw_question"`
}

type Data struct {
	TotalCredits float64 `json:"total_credits"`
	TotalUsage   float64 `json:"total_usage"`
}

type OpenRouteCredits struct {
	Data Data `json:"data"`
}

// DTOs para streaming

type OpenRouteStreamDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type OpenRouteStreamChoice struct {
	Index        int                  `json:"index"`
	Delta        OpenRouteStreamDelta `json:"delta"`
	FinishReason *string              `json:"finish_reason,omitempty"`
}

type OpenRouteStreamError struct {
	Code    interface{} `json:"code,omitempty"` // pode ser int ou string
	Message string      `json:"message,omitempty"`
}

type OpenRouteStreamChunk struct {
	ID       string                  `json:"id,omitempty"`
	Object   string                  `json:"object,omitempty"`
	Created  int64                   `json:"created,omitempty"`
	Model    string                  `json:"model,omitempty"`
	Provider string                  `json:"provider,omitempty"`
	Choices  []OpenRouteStreamChoice `json:"choices,omitempty"`
	Error    *OpenRouteStreamError   `json:"error,omitempty"`
	Usage    *UsageDTO               `json:"usage,omitempty"`
}

func NewClient(config *ClientConfig) *Client {
	if config == nil {
		config = DefaultConfig()
	}

	transport := &http.Transport{
		MaxIdleConns:        config.MaxIdleConns,
		MaxConnsPerHost:     config.MaxConnsPerHost,
		IdleConnTimeout:     90 * time.Second,
		DisableCompression:  false,
		DisableKeepAlives:   false,
		ForceAttemptHTTP2:   true,
	}

	return &Client{
		config: config,
		httpClient: &http.Client{
			Transport: transport,
			Timeout:   0, // Sem timeout global para streaming
		},
		bufferPool: NewBufferPool(4096),
		stats:      NewConnectionStats(),
	}
}

func (lw *OpenRouteWrapper) GetCurrentCredits() OpenRouteCredits {
	var credits OpenRouteCredits
	completionsAPI := fmt.Sprintf("%s/credits", OpenRouteApiUrl)
	ctx := context.Background()
	req, err := http.NewRequestWithContext(ctx, "GET", completionsAPI, nil)
	if err != nil {
		return OpenRouteCredits{
			Data: Data{
				TotalCredits: 0,
				TotalUsage:   0,
			},
		}
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", OpenRouteApiKey))
	resp, err := lw.httpClient.Do(req)
	if err != nil {
		return OpenRouteCredits{
			Data: Data{
				TotalCredits: 0,
				TotalUsage:   0,
			},
		}
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return OpenRouteCredits{
			Data: Data{
				TotalCredits: 0,
				TotalUsage:   0,
			},
		}
	}
	err = json.Unmarshal(body, &credits)
	if err != nil {
		return OpenRouteCredits{
			Data: Data{
				TotalCredits: 0,
				TotalUsage:   0,
			},
		}
	}
	return credits
}

func (lw *OpenRouteWrapper) CallOpenRouterAPI(openRouteDTO *OpenRoute) ([]*ResultService, error) {
	ctx := context.Background()
	completionsAPI := fmt.Sprintf("%s/chat/completions", OpenRouteApiUrl)

	jsonData, err := json.Marshal(&openRouteDTO)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, "POST", completionsAPI, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", OpenRouteApiKey))

	resp, err := lw.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}

	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("open router API error: status %d, body: %s", resp.StatusCode, string(body))
	}
	openRouteResponseDTO := new(OpenRouteResponseDTO)
	openRouteResponseDTO.Choices = make([]ResponseChoiceDTO, 0)
	if err := json.NewDecoder(resp.Body).Decode(&openRouteResponseDTO); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}
	resultServiceListDTO := make([]*ResultService, 0)
	for _, choice := range openRouteResponseDTO.Choices {
		jsonBytes, err := json.Marshal(choice)
		if err != nil {
			fmt.Printf("Erro ao fazer marshal: %v\n", err)
		}
		strJson := string(jsonBytes)
		strJson = strings.ReplaceAll(strJson, "```json", "")
		strJson = strings.ReplaceAll(strJson, "```", "")
		jsonBytes = []byte(strJson)
		var unmarshaled ResponseChoiceDTO
		err = json.Unmarshal(jsonBytes, &unmarshaled)
		if err != nil {
			fmt.Printf("Erro ao fazer unmarshal: %v\n", err)
		}
		errParse := ParseJSONFromAny(unmarshaled.Message.Content, &resultServiceListDTO)
		if errParse != nil {
			log.Fatal("Erro:", errParse)
		}
	}
	return resultServiceListDTO, nil
}

func ParseJSONFromAny[T any](data any, target *T) error {
	var jsonBytes []byte
	switch v := data.(type) {
	case string:
		cleaned := strings.TrimSpace(v)
		jsonBytes = []byte(cleaned)
	case []byte:
		jsonBytes = v
	default:
		var err error
		jsonBytes, err = json.Marshal(v)
		if err != nil {
			return fmt.Errorf("not possible to convert to JSON: %w", err)
		}
	}
	if !json.Valid(jsonBytes) {
		return fmt.Errorf("invalid JSON: %s", string(jsonBytes))
	}
	return json.Unmarshal(jsonBytes, target)
}

func ExtractJSONFromMarkdown(text string) (string, error) {

	re := regexp.MustCompile("(?s)```json\\s*\\n(.*?)\\n```")
	matches := re.FindStringSubmatch(text)

	if len(matches) < 2 {
		return "", fmt.Errorf("JSON não encontrado no texto")
	}

	return matches[1], nil
}

func (lw *OpenRouteWrapper) CallOpenRouterAPIStream(ctx context.Context, openRouteDTO *OpenRoute) (<-chan StreamEvent, error) {

	openRouteDTO.Stream = true

	completionsAPI := fmt.Sprintf("%s/chat/completions", OpenRouteApiUrl)

	jsonData, err := json.Marshal(&openRouteDTO)
	
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", completionsAPI, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", OpenRouteApiKey))
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("Cache-Control", "no-cache")
	req.Header.Set("Connection", "keep-alive")

	resp, err := lw.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("open router API error: status %d, body: %s", resp.StatusCode, string(body))
	}

	eventChan := make(chan StreamEvent, 100)

	go func() {
		defer close(eventChan)
		defer resp.Body.Close()

		reader := bufio.NewReader(resp.Body)

		for {
			select {
			case <-ctx.Done():
				eventChan <- StreamEvent{Error: ctx.Err()}
				return
			default:
			}

			line, err := reader.ReadString('\n')
			if err != nil {
				if err == io.EOF {
					eventChan <- StreamEvent{Done: true}
					return
				}
				eventChan <- StreamEvent{Error: fmt.Errorf("failed to read stream: %w", err)}
				return
			}

			line = strings.TrimSpace(line)

			if line == "" {
				continue
			}

			if strings.HasPrefix(line, ":") {
				continue
			}

			if !strings.HasPrefix(line, "data: ") {
				continue
			}

			data := strings.TrimPrefix(line, "data: ")

			if data == "[DONE]" {
				eventChan <- StreamEvent{Done: true}
				return
			}

			var chunk StreamChunk
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				eventChan <- StreamEvent{Error: fmt.Errorf("failed to parse chunk: %w, data: %s", err, data)}
				continue
			}

			if chunk.Error != nil {
				eventChan <- StreamEvent{
					Chunk: &chunk,
					Error: fmt.Errorf("stream error: %s", chunk.Error.Message),
				}
				return
			}

			eventChan <- StreamEvent{Chunk: &chunk}
		}
	}()

	return eventChan, nil
}

func (lw *OpenRouteWrapper) CallOpenRouterAPIStreamWithCallback(
	ctx context.Context,
	openRouteDTO *OpenRoute,
	onContent func(content string),
	onError func(err error),
	onDone func(fullContent string, usage *UsageDTO),
) error {
	eventChan, err := lw.CallOpenRouterAPIStream(ctx, openRouteDTO)
	if err != nil {
		return err
	}

	var fullContent strings.Builder
	var lastUsage *UsageDTO

	for event := range eventChan {
		if event.Error != nil {
			if onError != nil {
				onError(event.Error)
			}
			return event.Error
		}

		if event.Done {
			if onDone != nil {
				onDone(fullContent.String(), lastUsage)
			}
			return nil
		}

		if event.Chunk != nil {
			// Armazena usage se presente
			if event.Chunk.Usage != nil {
				lastUsage = event.Chunk.Usage
			}

			// Processa cada choice
			for _, choice := range event.Chunk.Choices {
				if choice.Delta.Content != "" {
					fullContent.WriteString(choice.Delta.Content)
					if onContent != nil {
						onContent(choice.Delta.Content)
					}
				}
			}
		}
	}

	return nil
}

func (c *Client) GetStats() map[string]int64 {
	return c.stats.GetStats()
}

func (c *Client) StreamCompletion(ctx context.Context, req *OpenRoute) (<-chan StreamEvent, error) {
	req.Stream = true

	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		fmt.Sprintf("%s/chat/completions", c.config.BaseURL),
		bytes.NewReader(jsonData),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.config.APIKey))
	httpReq.Header.Set("Accept", "text/event-stream")
	httpReq.Header.Set("Cache-Control", "no-cache")
	httpReq.Header.Set("Connection", "keep-alive")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		c.stats.IncrementErrors()
		return nil, fmt.Errorf("failed to send request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		c.stats.IncrementErrors()
		return nil, fmt.Errorf("API error: status %d, body: %s", resp.StatusCode, string(body))
	}

	c.stats.IncrementStreams()
	eventChan := make(chan StreamEvent, 64)

	go c.processStream(ctx, resp, eventChan)

	return eventChan, nil
}

func (c *Client) processStream(ctx context.Context, resp *http.Response, eventChan chan<- StreamEvent) {
	defer func() {
		close(eventChan)
		resp.Body.Close()
		c.stats.DecrementStreams()
	}()

	reader := bufio.NewReaderSize(resp.Body, 8192)

	for {
		select {
		case <-ctx.Done():
			eventChan <- StreamEvent{Error: ctx.Err()}
			return
		default:
		}

		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				eventChan <- StreamEvent{Done: true}
				return
			}
			c.stats.IncrementErrors()
			eventChan <- StreamEvent{Error: fmt.Errorf("read error: %w", err)}
			return
		}

		line = strings.TrimSpace(line)

		if line == "" || strings.HasPrefix(line, ":") {
			continue
		}

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")

		if data == "[DONE]" {
			eventChan <- StreamEvent{Done: true}
			return
		}

		var chunk StreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			c.stats.IncrementErrors()
			eventChan <- StreamEvent{Error: fmt.Errorf("parse error: %w", err)}
			continue
		}

		if chunk.Error != nil {
			c.stats.IncrementErrors()
			eventChan <- StreamEvent{
				Chunk: &chunk,
				Error: fmt.Errorf("stream error: %s", chunk.Error.Message),
			}
			return
		}

		for _, choice := range chunk.Choices {
			if choice.Delta.Content != "" {
				c.stats.AddTokens(1)
			}
		}

		eventChan <- StreamEvent{Chunk: &chunk}
	}
}

func (c *Client) Close() error {
	c.httpClient.CloseIdleConnections()
	return nil
}

