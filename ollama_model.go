package main

import "time"

type ChatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatResponse struct {
	Model     string    `json:"model"`
	Message   Message   `json:"message"`
	CreatedAt time.Time `json:"created_at"`
	Done      bool      `json:"done"`
}

type EmbeddingRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type EmbeddingResponse struct {
	Embedding []float64 `json:"embedding"`
}

type Document struct {
	ID        string    `json:"id"`
	Content   string    `json:"content"`
	Embedding []float64 `json:"embedding"`
	Metadata  map[string]interface{} `json:"metadata"`
	CreatedAt time.Time `json:"created_at"`
}

type IndexRequest struct {
	Content  string                 `json:"content"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

type SearchRequest struct {
	Query string `json:"query"`
	Limit int    `json:"limit"`
}

type SearchResult struct {
	Document   Document `json:"document"`
	Similarity float64  `json:"similarity"`
}

type SearchResponse struct {
	Results []SearchResult `json:"results"`
	Took    string         `json:"took"`
}

type GenerateRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream"`
}

type GenerateResponse struct {
	Model     string    `json:"model"`
	Response  string    `json:"response"`
	CreatedAt time.Time `json:"created_at"`
	Done      bool      `json:"done"`
}

type ErrorResponse struct {
	Error   string `json:"error"`
	Message string `json:"message"`
}