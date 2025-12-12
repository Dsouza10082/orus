package main

import "time"

type ChatRequest struct {
	Model    string    `json:"model" swaggertype:"string" example:"llama3.1:8b"`
	Messages []Message `json:"messages" swaggertype:"array" example:"[{role: 'user', content: 'Hello, how are you?'}]"`
	Stream   bool      `json:"stream" swaggertype:"boolean" example:"true"`
	Format   string    `json:"format" swaggertype:"string" example:"json"`
	Think    bool      `json:"think" swaggertype:"boolean" example:"true"`
}

type Message struct {
	Role    string `json:"role" swaggertype:"string" example:"user"`
	Content string `json:"content" swaggertype:"string" example:"Hello, how are you?"`
}

type ChatResponse struct {
	Model     string    `json:"model" swaggertype:"string" example:"llama3.1:8b"`
	Message   Message   `json:"message" swaggertype:"object" example:"{role: 'user', content: 'Hello, how are you?'}"`
	CreatedAt time.Time `json:"created_at" swaggertype:"object"`
	Done      bool      `json:"done" swaggertype:"boolean" example:"true"`
}

type EmbeddingRequest struct {
	Model  string `json:"model" swaggertype:"string" example:"llama3.1:8b"`
	Prompt string `json:"prompt" swaggertype:"string" example:"Hello, how are you?"`
}

type EmbeddingResponse struct {
	Embedding []float64 `json:"embedding" swaggertype:"array" example:"[0.1, 0.2, 0.3]"`
}

type Document struct {
	ID        string                 `json:"id" swaggertype:"string" example:"123e4567-e89b-12d3-a456-426614174000"`
	Content   string                 `json:"content" swaggertype:"string" example:"Hello, how are you?"`
	Embedding []float64              `json:"embedding" swaggertype:"array" example:"[0.1, 0.2, 0.3]"`
	Metadata  map[string]interface{} `json:"metadata" swaggertype:"object"`
	CreatedAt time.Time              `json:"created_at" swaggertype:"string" example:"2021-01-01T00:00:00Z"`
}

type IndexRequest struct {
	Content  string                 `json:"content"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

type SearchRequest struct {
	Query string `json:"query" swaggertype:"string" example:"Hello, how are you?"`
	Limit int    `json:"limit" swaggertype:"integer" example:"10"`
}

type SearchResult struct {
	Document   Document `json:"document" swaggertype:"object"`
	Similarity float64  `json:"similarity" swaggertype:"number" example:"0.95"`
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
