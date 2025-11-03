package orus

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
)

type OrusAPI struct {
	*Orus
	Port   string
	router *chi.Mux
	Verbose bool
}

func NewOrusAPI() *OrusAPI {
	return &OrusAPI{
		Orus:   NewOrus(),
		Port:   LoadEnv("API_PORT"),
		router: chi.NewRouter(),
		Verbose: false,
	}
}

func (s *OrusAPI) setupRoutes() {
	s.router.Get("/orus-api/v1/system-info", s.GetSystemInfo)
	s.router.Post("/orus-api/v1/bge-m3-embed", s.BGE_M3_Embed)
	s.router.Post("/orus-api/v1/ollama-embed", s.OllamaEmbed)
	s.router.Get("/orus-api/v1/ollama-model-list", s.OllamaModelList)
}

func (s *OrusAPI) Start() {
	s.setupRoutes()
	log.Printf("Orus API server started on port %s", s.Port)
	err := http.ListenAndServe(":"+s.Port, s.router)
	if err != nil {
		log.Fatalf("Failed to start server: %v", err)
		return
	}
}

func (s *OrusAPI) SetVerbose(verbose bool) *OrusAPI {
	s.Verbose = verbose
	return s
}


func (s *OrusAPI) GetSystemInfo(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()
	response := OrusResponse{
		Success:   true,
		Serial:    uuid.New().String(),
		Message:   "System info retrieved successfully",
		Error:     "",
		TimeTaken: time.Since(startTime),
		Data: map[string]interface{}{
			"version":     "1.0.0",
			"name":        "Orus",
			"description": "Orus is a server for the Orus library",
			"author":      "Dsouza10082",
			"author_url":  "https://github.com/Dsouza10082",
		},
	}
	respondJSON(w, http.StatusOK, response)
}

func (s *OrusAPI) BGE_M3_Embed(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	response := NewOrusResponse()
	request := new(OrusRequest)
	json.Unmarshal(body, request)
	respChan := make(chan *OrusResponse)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	go func() {
		select {
		case <-ctx.Done():
			return
		default:
			vector, err := s.BGEM3Embedder.Embed(request.Body["text"].(string))
			if err != nil {
				response.Error = err.Error()
				response.Success = false
				response.TimeTaken = time.Since(startTime)
				respChan <- response
			}
			response.Data = map[string]interface{}{
				"vector": vector,
			}
			response.Success = true
			response.TimeTaken = time.Since(startTime)
			response.Message = "Embed request received successfully"
			respChan <- response
		}
	}()
	select {
	case response := <-respChan:
		respondJSON(w, http.StatusOK, response)
	case <-ctx.Done():
		response.Error = "Error Timeout"
		response.Success = false
		response.TimeTaken = time.Since(startTime)
		respondJSON(w, http.StatusInternalServerError, response)
	}
}

func (s *OrusAPI) OllamaEmbed(w http.ResponseWriter, r *http.Request) {
	
}

func (s *OrusAPI) OllamaModelList(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()
	models, err := s.OllamaClient.ListModels()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	response := NewOrusResponse()
	response.Data = map[string]interface{}{
		"models": models,
	}
	response.Success = true
	response.TimeTaken = time.Since(startTime)
	response.Message = "Ollama model list retrieved successfully"
	respondJSON(w, http.StatusOK, response)
}

