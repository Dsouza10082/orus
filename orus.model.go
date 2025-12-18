package main

import (
	"encoding/json"
	"log/slog"
	"net/http"
	"time"

	"github.com/google/uuid"
)

type LLMCloudRequestBody struct {
	Model    string    `json:"model"`
	Think    bool      `json:"think"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream"`
	Format   string    `json:"format,omitempty"`
	Images   []string  `json:"images,omitempty"`
}

type LLMCloudRequest struct {
	Created string              `json:"created"`
	Body    LLMCloudRequestBody `json:"body"`
}



type OrusResponse struct {
	Success   bool                   `json:"success" swaggertype:"boolean" example:"true"`
	Serial    string                 `json:"serial" swaggertype:"string" example:"123e4567-e89b-12d3-a456-426614174000"`
	Message   string                 `json:"message" swaggertype:"string" example:"Request received successfully"`
	Data      map[string]interface{} `json:"data" swaggertype:"object" `
	Error     string                 `json:"error" swaggertype:"string" example:"Error message"`
	TimeTaken time.Duration          `json:"time_taken" swaggertype:"integer" example:"1500"`
}

func NewOrusResponse() *OrusResponse {
	return &OrusResponse{
		Success: true,
		Serial: uuid.New().String(),
		Message: "Request received successfully",
		Data: make(map[string]interface{}),
		Error: "",
	}
}

type OrusRequest struct {
	Created string                 `json:"created" swaggertype:"string" example:"2021-01-01T00:00:00Z"`
	Body    map[string]interface{} `json:"body" swaggertype:"interface{}" example:"{"text": "Hello, how are you?"}`
}

func releaseChatRequest(chatRequest *ChatRequest) {
	chatRequest.Messages = chatRequest.Messages[:0]
	chatRequest.Images = chatRequest.Images[:0]
	chatRequest.Format = ""
	chatRequest.Model = ""
	chatRequestPool.Put(chatRequest)
}

func logRequest(requestID string, chatRequest *ChatRequest) {
	slog.Debug("LLM request",
		"request_id", requestID,
		"model", chatRequest.Model,
		"think", chatRequest.Think,
		"stream", chatRequest.Stream,
		"messages_count", len(chatRequest.Messages),
	)
}


func respondJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func respondError(w http.ResponseWriter, status int, code, message string) {
	respondJSON(w, status, map[string]interface{}{
		"success": false,
		"error":   code,
		"message": message,
	})
}