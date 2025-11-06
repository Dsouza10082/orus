package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/google/uuid"
)

type OrusAPI struct {
	*Orus
	Port    string
	router  *chi.Mux
	Verbose bool
}

func NewOrusAPI() *OrusAPI {
	router := chi.NewRouter()
	router.Use(middleware.Logger)
	router.Use(middleware.Recoverer)
	router.Use(middleware.StripSlashes)
	router.Use(middleware.URLFormat)

	return &OrusAPI{
		Orus:    NewOrus(),
		Port:    LoadEnv("ORUS_API_PORT"),
		router:  router,
		Verbose: false,
	}
}


func (s *OrusAPI) setupRoutes() {
	s.router.Get("/orus-api/v1/system-info", s.GetSystemInfo)
	s.router.Post("/orus-api/v1/embed-text", s.EmbedText)
	s.router.Get("/orus-api/v1/ollama-model-list", s.OllamaModelList)
	s.router.Post("/orus-api/v1/ollama-pull-model", s.OllamaPullModel)
	s.router.Post("/orus-api/v1/call-llm", s.CallLLM)
}

func (s *OrusAPI) Start() {
	s.setupRoutes()
	log.Println("Orus API ORUS_API_AGENT_MEMORY_PATH", LoadEnv("ORUS_API_AGENT_MEMORY_PATH"))
	log.Println("Orus API ORUS_API_TOK_PATH", LoadEnv("ORUS_API_TOK_PATH"))
	log.Println("Orus API ORUS_API_ONNX_PATH", LoadEnv("ORUS_API_ONNX_PATH"))
	log.Println("Orus API ORUS_API_ONNX_RUNTIME_PATH", LoadEnv("ORUS_API_ONNX_RUNTIME_PATH"))
	log.Println("Orus API ORUS_API_OLLAMA_BASE_URL", LoadEnv("ORUS_API_OLLAMA_BASE_URL"))

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

func (s *OrusAPI) EmbedText(w http.ResponseWriter, r *http.Request) {
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
			model := request.Body["model"].(string)
			text := request.Body["text"].(string)
			response := s.embedText(model, text, startTime)
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
		response.Message = "Error Timeout"
		respondJSON(w, http.StatusInternalServerError, response)
	}
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
			vector, err := s.Orus.BGEM3Embedder.Embed(request.Body["text"].(string))
			if err != nil {
				response.Error = err.Error()
				response.Success = false
				response.Message = "Error embedding text with BGE-M3"
				response.TimeTaken = time.Since(startTime)
				respChan <- response
			} else {
				response.Data = map[string]interface{}{
					"vector": vector,
				}
				response.Success = true
				response.TimeTaken = time.Since(startTime)
				response.Message = "Embed request received successfully"
				respChan <- response
			}
		}
	}()
	select {
	case response := <-respChan:
		respondJSON(w, http.StatusOK, response)
	case <-ctx.Done():
		response.Error = "Error Timeout"
		response.Success = false
		response.TimeTaken = time.Since(startTime)
		response.Message = "Error Timeout"
		respondJSON(w, http.StatusInternalServerError, response)
	}
}

func (s *OrusAPI) OllamaEmbed(w http.ResponseWriter, r *http.Request) {
	
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
			model := request.Body["model"].(string)
			text := request.Body["text"].(string)
			vector, err := s.OllamaClient.GetEmbedding(model, text)
			if err != nil {
				response.Error = err.Error()
				response.Success = false
				response.TimeTaken = time.Since(startTime)
				response.Message = "Error embedding text with Ollama"
				respChan <- response
			} else {
				response.Data = map[string]interface{}{
					"vector": vector,
				}
				response.Success = true
				response.TimeTaken = time.Since(startTime)
				response.Message = "Embed request received successfully"
				respChan <- response
			}
		}
	}()
	select {
	case response := <-respChan:
		respondJSON(w, http.StatusOK, response)
	case <-ctx.Done():
		response.Error = "Error Timeout"
		response.Success = false
		response.TimeTaken = time.Since(startTime)
		response.Message = "Error Timeout"
		respondJSON(w, http.StatusInternalServerError, response)
	}
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

func (s *OrusAPI) OllamaPullModel(w http.ResponseWriter, r *http.Request) {
	
	var req struct {
		Name string `json:"name"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, http.StatusBadRequest, "invalid_request", "Invalid JSON format")
		return
	}

	if req.Name == "" {
		respondError(w, http.StatusBadRequest, "missing_name", "Field 'name' is required")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	
	flusher, ok := w.(http.Flusher)
	if !ok {
		respondError(w, http.StatusInternalServerError, "streaming_not_supported", "Streaming not supported")
		return
	}

	progressCallback := func(progress PullModelProgress) {
		data, _ := json.Marshal(progress)
		fmt.Fprintf(w, "data: %s\n\n", string(data))
		flusher.Flush()
	}

	err := s.OllamaClient.PullModel(req.Name, progressCallback)
	if err != nil {
		errorData, _ := json.Marshal(map[string]string{
			"status": "error",
			"error":  err.Error(),
		})
		fmt.Fprintf(w, "data: %s\n\n", string(errorData))
		flusher.Flush()
		return
	}

	successData, _ := json.Marshal(map[string]string{
		"status":  "success",
		"message": fmt.Sprintf("Model %s downloaded successfully", req.Name),
	})
	fmt.Fprintf(w, "data: %s\n\n", string(successData))
	flusher.Flush()
}


func (s *OrusAPI) embedText(model string, text string, startTime time.Time) *OrusResponse {
	response := NewOrusResponse()
	switch model {
	case "bge-m3":
		vector, err := s.Orus.BGEM3Embedder.Embed(text)
		if err != nil {
			response.Error = err.Error()
			response.Success = false
			response.TimeTaken = time.Since(startTime)
			response.Message = "Error embedding text with BGE-M3"
			return response
		}
		response.Data = map[string]interface{}{
			"vector": vector,
			"text": text,
			"model": "bge-m3",
			"dimensions": len(vector),
		}
		response.Success = true
		response.TimeTaken = time.Since(startTime)
		response.Message = "Embed request received successfully"
		return response
	case "nomic-embed-text":
		vector, err := s.Orus.OllamaClient.GetEmbedding(model, text)
		if err != nil {
			response.Error = err.Error()
			response.Success = false
			response.TimeTaken = time.Since(startTime)
			return response
		}
		response.Data = map[string]interface{}{
			"vector": vector,
			"text": text,
			"model": "nomic-embed-text",
			"dimensions": len(vector),
		}
		response.Success = true
		response.TimeTaken = time.Since(startTime)
		response.Message = "Embed request received successfully"
		return response
	case "ollama":
		vector, err := s.Orus.OllamaClient.GetEmbedding(model, text)
		if err != nil {
			response.Error = err.Error()
			response.Success = false
			response.TimeTaken = time.Since(startTime)
			return response
		}
		response.Data = map[string]interface{}{
			"vector": vector,
			"text": text,
			"model": "ollama",
			"dimensions": len(vector),
		}
		response.Success = true
		response.TimeTaken = time.Since(startTime)
		response.Message = "Embed request received successfully"
		return response
	default:
		response.Error = "Invalid model"
		response.Success = false
		response.TimeTaken = time.Since(startTime)
		response.Message = "Invalid model"
		return response
	}
}

func (s *OrusAPI) CallLLM(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()
	body, err := io.ReadAll(r.Body)

	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	response := NewOrusResponse()
	request := new(OrusRequest)
	json.Unmarshal(body, request)
	
	model := request.Body["model"].(string)
	messagesRaw := request.Body["messages"]
	messagesJSON, _ := json.Marshal(messagesRaw)
	
	var messages []Message
	if err := json.Unmarshal(messagesJSON, &messages); err != nil {
		response.Error = err.Error()
		response.Message = "Error unmarshalling messages"
		response.Success = false
		response.TimeTaken = time.Since(startTime)
		respondJSON(w, http.StatusBadRequest, response)
	}

	stream := request.Body["stream"].(bool)

	if stream {

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
        
		content := make([]string, 0)
		flusher, ok := w.(http.Flusher)
		if !ok {
			respondError(w, http.StatusInternalServerError, "streaming_not_supported", "Streaming not supported")
			return
		}
		flusher.Flush()
		chatStreamProgressCallback := func(chatResp ChatStreamResponse) {
			data, _ := json.Marshal(chatResp)
			fmt.Fprintf(w, "data: %s\n\n", string(data))
			flusher.Flush()
			content = append(content, chatResp.Message.Content)
		}
		err := s.OllamaClient.ChatStream(ChatRequest{
			Model: model,
			Messages: messages,
			Stream: stream,
		}, chatStreamProgressCallback)
		if err != nil {
			errorData, _ := json.Marshal(map[string]string{
				"status": "error",
				"error":  err.Error(),
			})
			fmt.Fprintf(w, "data: %s\n\n", string(errorData))
			flusher.Flush()
			return
		}
		successData, _ := json.Marshal(map[string]string{
			"status": "success",
			"message": "LLM request received successfully",
			"content": strings.Join(content, ""),
		})
		fmt.Fprintf(w, "data: %s\n\n", string(successData))
		flusher.Flush()
		return
	} else {
		responseLLM, err := s.OllamaClient.Chat(ChatRequest{
			Model: model,
			Messages: messages,
			Stream: stream,
		})
		if err != nil {
			response.Error = err.Error()
			response.Message = "Error calling LLM"
			response.Success = false
			response.TimeTaken = time.Since(startTime)
			respondJSON(w, http.StatusInternalServerError, response)
		} else {
			response.Data = map[string]interface{}{
				"response": responseLLM.Message.Content,
				"model": model,
				"messages": messages,
				"stream": stream,
			}
			response.Success = true
			response.TimeTaken = time.Since(startTime)
			response.Message = "LLM request received successfully"
			respondJSON(w, http.StatusOK, response)
		}
	}			
}

// ---------------------------MAIN FUNCTION------------------------------

func main() {
	orusApi := NewOrusAPI()
	orusApi.Start()
}