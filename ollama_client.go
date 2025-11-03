package orus

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
	httpClient *http.Client
}

func NewOllamaClient(baseURL string) *OllamaClient {
	return &OllamaClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 300 * time.Second,
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