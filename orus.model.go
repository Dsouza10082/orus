package orus

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/google/uuid"
)

type OrusResponse struct {
	Success   bool                   `json:"success"`
	Serial    string                 `json:"serial"`
	Message   string                 `json:"message"`
	Data      map[string]interface{} `json:"data"`
	Error     string                 `json:"error"`
	TimeTaken time.Duration          `json:"time_taken"`
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
	Created string                 `json:"created"`
	Body    map[string]interface{} `json:"body"`
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