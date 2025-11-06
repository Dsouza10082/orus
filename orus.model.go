package main

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/google/uuid"
)

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

func respondJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func respondError(w http.ResponseWriter, status int, errorType, message string) {
	respondJSON(w, status, ErrorResponse{
		Error:   errorType,
		Message: message,
	})
}