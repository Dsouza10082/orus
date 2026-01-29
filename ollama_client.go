package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type OllamaClient struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
}

type WebSearchResult struct {
	Title   string `json:"title"`
	URL     string `json:"url"`
	Content string `json:"content"`
}

type WebSearchResponse struct {
	Results []WebSearchResult `json:"results"`
}

type PullModelProgress struct {
	Status    string `json:"status"`
	Digest    string `json:"digest,omitempty"`
	Total     int64  `json:"total,omitempty"`
	Completed int64  `json:"completed,omitempty"`
}

type ChatStreamResponse struct {
	Model     string    `json:"model"`
	Message   Message   `json:"message"`
	CreatedAt time.Time `json:"created_at"`
	Done      bool      `json:"done"`
	Progress  int       `json:"progress"`
	Total     int64     `json:"total,omitempty"`
	Completed int64     `json:"completed,omitempty"`
}

type WebSearchRequest struct {
	Query string `json:"query"`
	MaxResults int `json:"max_results"`
}

func NewOllamaClient() *OllamaClient {
	apiKey := LoadEnv("OLLAMA_API_KEY")
	baseURL := LoadEnv("ORUS_API_OLLAMA_BASE_URL")
	return &OllamaClient{
		baseURL: baseURL,
		apiKey: apiKey,
		httpClient: &http.Client{
			Timeout: 2000 * time.Second,
		},
	}
}

func (c *OllamaClient) Generate(req GenerateRequest) (*GenerateResponse, error) {
	url := fmt.Sprintf("%s/api/generate", c.baseURL)

	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("error serializing request: %w", err)
	}

	httpReq, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("error creating request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("error making request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("error from Ollama (status %d): %s", resp.StatusCode, string(body))
	}

	decoder := json.NewDecoder(resp.Body)
	var finalResponse GenerateResponse

	for decoder.More() {
		var genResp GenerateResponse
		if err := decoder.Decode(&genResp); err != nil {
			return nil, fmt.Errorf("error decoding response: %w", err)
		}
		finalResponse.Response += genResp.Response
		finalResponse.Model = genResp.Model
		finalResponse.CreatedAt = genResp.CreatedAt
		finalResponse.Done = genResp.Done

		if genResp.Done {
			break
		}
	}

	return &finalResponse, nil
}

func (c *OllamaClient) Chat(req ChatRequest) (*ChatResponse, error) {
	url := fmt.Sprintf("%s/api/chat", c.baseURL)
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("error serializing request: %w", err)
	}

	httpReq, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("error creating request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("error making request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("error from Ollama (status %d): %s", resp.StatusCode, string(body))
	}
	decoder := json.NewDecoder(resp.Body)
	var finalResponse ChatResponse
	var fullContent string
	for decoder.More() {
		var chatResp ChatResponse
		if err := decoder.Decode(&chatResp); err != nil {
			return nil, fmt.Errorf("error decoding response: %w", err)
		}
		fullContent += chatResp.Message.Content
		finalResponse.Model = chatResp.Model
		finalResponse.CreatedAt = chatResp.CreatedAt
		finalResponse.Done = chatResp.Done
		finalResponse.Message.Role = chatResp.Message.Role

		if chatResp.Done {
			break
		}
	}
	finalResponse.Message.Content = fullContent
	return &finalResponse, nil
}

func (c *OllamaClient) ChatCloud(req ChatRequest) (*ChatResponse, error) {
	
	url := "https://ollama.com/api/chat"
	
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("error serializing request: %w", err)
	}
	httpReq, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("error creating request: %w", err)
	}
	httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiKey))
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("error making request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("error from Ollama (status %d): %s", resp.StatusCode, string(body))
	}
	decoder := json.NewDecoder(resp.Body)
	var finalResponse ChatResponse
	var fullContent string
	for decoder.More() {
		var chatResp ChatResponse
		if err := decoder.Decode(&chatResp); err != nil {
			return nil, fmt.Errorf("error decoding response: %w", err)
		}
		fullContent += chatResp.Message.Content
		finalResponse.Model = chatResp.Model
		finalResponse.CreatedAt = chatResp.CreatedAt
		finalResponse.Done = chatResp.Done
		finalResponse.Message.Role = chatResp.Message.Role

		if chatResp.Done {
			break
		}
	}
	finalResponse.Message.Content = fullContent
	return &finalResponse, nil
}

func (c *OllamaClient) ChatStream(req ChatRequest, chatStreamProgressCallback func(ChatStreamResponse)) error {
	req.Stream = true
	url := fmt.Sprintf("%s/api/chat", c.baseURL)
	jsonData, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("error serializing request: %w", err)
	}
	httpReq, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("error creating request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("error making request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("error from Ollama (status %d): %s", resp.StatusCode, string(body))
	}
	decoder := json.NewDecoder(resp.Body)
	for decoder.More() {
		var chatResp ChatStreamResponse
		var fullContent string
		if err := decoder.Decode(&chatResp); err != nil {
			return fmt.Errorf("error decoding response: %w", err)
		}
		fullContent += chatResp.Message.Content
		chatResp.Message.Content = fullContent
		chatResp.Model = req.Model
		chatResp.CreatedAt = time.Now()
		chatStreamProgressCallback(chatResp)
		if chatResp.Done {
			break
		}
	}
	return nil
}

func (c *OllamaClient) ChatStreamCloud(req ChatRequest, chatStreamProgressCallback func(ChatStreamResponse)) error {
	req.Stream = true
	url := "https://ollama.com/api/chat"
	jsonData, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("error serializing request: %w", err)
	}
	httpReq, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("error creating request: %w", err)
	}
	httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiKey))
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("error making request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("error from Ollama (status %d): %s", resp.StatusCode, string(body))
	}
	decoder := json.NewDecoder(resp.Body)
	for decoder.More() {
		var chatResp ChatStreamResponse
		var fullContent string
		if err := decoder.Decode(&chatResp); err != nil {
			return fmt.Errorf("error decoding response: %w", err)
		}
		fullContent += chatResp.Message.Content
		chatResp.Message.Content = fullContent
		chatResp.Model = req.Model
		chatResp.CreatedAt = time.Now()
		chatStreamProgressCallback(chatResp)
		if chatResp.Done {
			break
		}
	}
	return nil
}

// GetEmbedding obtém embeddings de um texto
func (c *OllamaClient) GetEmbedding(model, text string) ([]float64, error) {
	url := fmt.Sprintf("%s/api/embeddings", c.baseURL)

	reqData := map[string]string{
		"model":  model,
		"prompt": text,
	}

	jsonData, err := json.Marshal(reqData)
	if err != nil {
		return nil, fmt.Errorf("error serializing request: %w", err)
	}

	httpReq, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("error creating request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("error making request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("error from Ollama (status %d): %s", resp.StatusCode, string(body))
	}

	var embResp EmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&embResp); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return embResp.Embedding, nil
}

// ListModels lista modelos disponíveis
func (c *OllamaClient) ListModels() ([]string, error) {
	url := fmt.Sprintf("%s/api/tags", c.baseURL)

	resp, err := c.httpClient.Get(url)
	if err != nil {
		return nil, fmt.Errorf("error making request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("error from Ollama (status %d)", resp.StatusCode)
	}

	var result struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	models := make([]string, len(result.Models))
	for i, m := range result.Models {
		models[i] = m.Name
	}

	return models, nil
}

func (c *OllamaClient) PullModel(modelName string, progressCallback func(PullModelProgress)) error {
	url := fmt.Sprintf("%s/api/pull", c.baseURL)

	reqData := map[string]interface{}{
		"name":   modelName,
		"stream": true,
	}

	jsonData, _ := json.Marshal(reqData)
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("erro do servidor: status %d", resp.StatusCode)
	}

	decoder := json.NewDecoder(resp.Body)
	for decoder.More() {
		var progress PullModelProgress
		if err := decoder.Decode(&progress); err != nil {
			return err
		}

		if progressCallback != nil {
			progressCallback(progress)
		}

		if progress.Status == "success" {
			break
		}
	}

	return nil
}

func (c *OllamaClient) WebSearch(query string, maxResults int) (*WebSearchResponse, error) {
	url := "https://ollama.com/api/web_search"


	if maxResults <= 0 {
		maxResults = 5
	}
	if maxResults > 10 {
		maxResults = 10
	}

	reqData := map[string]interface{}{
		"query":       query,
		"max_results": maxResults,
	}

	jsonData, _ := json.Marshal(reqData)
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiKey))

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("erro do servidor: status %d - %s", resp.StatusCode, string(body))
	}

	var searchResponse WebSearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&searchResponse); err != nil {
		return nil, fmt.Errorf("erro ao decodificar resposta: %w", err)
	}

	return &searchResponse, nil
}