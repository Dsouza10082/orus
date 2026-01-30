package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	view "github.com/Dsouza10082/orus/view"
	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/google/uuid"
	"github.com/starfederation/datastar-go/datastar"
	httpSwagger "github.com/swaggo/http-swagger"

	_ "github.com/Dsouza10082/orus/docs"
)

type OrusAPI struct {
	*Orus
	Port          string
	router        *chi.Mux
	Verbose       bool
	server        *http.Server
	activeConns   map[string]context.CancelFunc
	activeConnsMu sync.RWMutex
}

type PromptSignals struct {
	Prompt        string `json:"prompt"`
	Model         string `json:"model"`
	OperationType string `json:"operationType"`
	ResponseMode  string `json:"responseMode"`
	Result        string `json:"result"`
}

var (
	bufferPool = sync.Pool{
		New: func() interface{} {
			return bytes.NewBuffer(make([]byte, 0, 4096))
		},
	}

	stringBuilderPool = sync.Pool{
		New: func() interface{} {
			return &strings.Builder{}
		},
	}

	chatRequestPool = sync.Pool{
		New: func() interface{} {
			return &ChatRequest{
				Messages: make([]Message, 0, 10),
				Images:   make([]string, 0, 4),
			}
		},
	}
)

// ==================== Configuration ====================

const (
	MaxBodySize      = 10 * 1024 * 1024 // 10MB
	MaxConcurrent    = 100
	RequestTimeout   = 5 * time.Minute
	StreamBufferSize = 32 * 1024
)

// ==================== Validation ====================

type ValidationError struct {
	Code    string
	Message string
}

func (e *ValidationError) Error() string {
	return e.Message
}

func validateLLMRequest(body *LLMCloudRequestBody) *ValidationError {
	if body.Model == "" {
		return &ValidationError{"missing_model", "Field 'model' is required"}
	}
	if len(body.Messages) == 0 {
		return &ValidationError{"missing_messages", "Field 'messages' is required"}
	}
	return nil
}

func NewOrusAPI() *OrusAPI {
	router := chi.NewRouter()
	router.Use(middleware.Logger)
	router.Use(middleware.Recoverer)
	router.Use(middleware.StripSlashes)
	router.Use(middleware.URLFormat)

	router.Use(middleware.RequestID)
	router.Use(middleware.RealIP)
	router.Use(RequestLogger)
	router.Use(middleware.Timeout(RequestTimeout))

	router.Use(func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			r.Body = http.MaxBytesReader(w, r.Body, MaxBodySize)
			next.ServeHTTP(w, r)
		})
	})

	server := &http.Server{
		Addr:              ":" + LoadEnv("ORUS_API_PORT"),
		Handler:           router,
		ReadTimeout:       0,
		WriteTimeout:      0,
		IdleTimeout:       120 * time.Second,
		ReadHeaderTimeout: 10 * time.Second,
		MaxHeaderBytes:    1 << 20,
	}
	return &OrusAPI{
		Orus:    NewOrus(),
		Port:    LoadEnv("ORUS_API_PORT"),
		router:  router,
		Verbose: false,
		server:  server,
	}
}

func DefaultOpenRouteConfig() *OpenRouteConfig {
	return &OpenRouteConfig{
		APIKey:          "",
		OpenRouteAPIKey: "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",
		MaxConnections:  1000,
		RequestTimeout:  30 * time.Second,
		EnableAuth:      false,
	}
}

func (s *OrusAPI) setupRoutes() {
	s.router.Get("/orus-api/v1/system-info", s.GetSystemInfo)
	s.router.Post("/orus-api/v1/embed-text", s.EmbedText)
	s.router.Get("/orus-api/v1/ollama-model-list", s.OllamaModelList)
	s.router.Post("/orus-api/v1/ollama-pull-model", s.OllamaPullModel)
	s.router.Post("/orus-api/v1/call-llm", s.CallLLM)
	s.router.Post("/orus-api/v1/call-llm-cloud", s.CallLLMCloud)
	s.router.Post("/orus-api/v2/call-llm", s.CallLLMOptimized)
	s.router.Post("/orus-api/v2/health-check", s.HealthCheck)
	s.router.Get("/prompt", s.IndexHandler)
	s.router.Post("/prompt/llm-stream", s.PromptLLMStream)
	s.router.Post("/orus-api/v2/openroute", s.HandleOpenRouteChatStream)
	s.router.Get("/orus-api/v2/openroute-credit", s.HandleOpenRouteChatCredit)
	s.router.Post("/orus-api/v2/web-search", s.HandleWebSearch)

	s.router.Get("/swagger/*", httpSwagger.Handler(
		httpSwagger.URL(fmt.Sprintf("http://localhost:%s/swagger/doc.json", s.Port)),
	))

	s.router.Get("/swagger/doc.json", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, "docs/swagger.json")
	})
}

// IndexHandler is a handler for the prompt endpoint
// It renders the index.html file
func (s *OrusAPI) IndexHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html")
	indexView := view.NewView()
	models, err := s.OllamaClient.ListModels()
	if err != nil {
		log.Printf("IndexHandler: failed to list models: %v", err)
		http.Error(w, "failed to list models", http.StatusInternalServerError)
		return
	}
	indexView.SetModels(models).
		RenderIndex(w)
}

// PromptLLMStream is a handler for the prompt/llm-stream endpoint
// It reads the signals from the request and sends them to the LLM
// It then streams the response back to the client
func (s *OrusAPI) PromptLLMStream(w http.ResponseWriter, r *http.Request) {

	signals := &PromptSignals{}
	if err := datastar.ReadSignals(r, signals); err != nil {
		log.Printf("PromptLLMStream: failed to read signals: %v", err)
		http.Error(w, "failed to read signals", http.StatusBadRequest)
		return
	}

	sse := datastar.NewSSE(w, r)

	if signals.OperationType == "embedding" {

		if signals.Model == "nomic-embed-text:latest" {
			embedding, err := s.OllamaClient.GetEmbedding(signals.Model, signals.Prompt)
			if err != nil {
				_ = sse.ConsoleError(fmt.Errorf("embedding error: %w", err))
				return
			}
			signals.Result = fmt.Sprintf("Nomic Embedding (768 dimensions): %v", embedding)
		} else {
			embedding, err := s.Orus.BGEM3Embedder.Embed(signals.Prompt)
			if err != nil {
				_ = sse.ConsoleError(fmt.Errorf("embedding error: %w", err))
				return
			}
			signals.Result = fmt.Sprintf("BGE-M3 Embedding (1024 dimensions): %v", embedding)
		}

		if err := sse.MarshalAndPatchSignals(signals); err != nil {
			_ = sse.ConsoleError(fmt.Errorf("failed to patch signals: %w", err))
		}
		return
	}

	if signals.Model == "" {
		signals.Model = "llama3.1:8b"
	}

	signals.ResponseMode = "stream"

	signals.Result = ""
	if err := sse.MarshalAndPatchSignals(signals); err != nil {
		_ = sse.ConsoleError(fmt.Errorf("failed to clear result: %w", err))
		return
	}

	messages := []Message{
		{
			Role:    "user",
			Content: signals.Prompt,
		},
	}

	if signals.ResponseMode == "single" {
		resp, err := s.OllamaClient.Chat(ChatRequest{
			Model:    signals.Model,
			Messages: messages,
			Stream:   false,
		})
		if err != nil {
			_ = sse.ConsoleError(fmt.Errorf("LLM error: %w", err))
			return
		}

		signals.Result = resp.Message.Content
		if err := sse.MarshalAndPatchSignals(signals); err != nil {
			_ = sse.ConsoleError(fmt.Errorf("failed to patch signals: %w", err))
		}
		return
	}

	err := s.OllamaClient.ChatStream(ChatRequest{
		Model:    signals.Model,
		Messages: messages,
		Stream:   true,
	}, func(chunk ChatStreamResponse) {
		if sse.IsClosed() {
			return
		}
		if chunk.Message.Content == "" {
			return
		}
		signals.Result += chunk.Message.Content
		if err := sse.MarshalAndPatchSignals(signals); err != nil {
			_ = sse.ConsoleError(fmt.Errorf("failed to patch signals: %w", err))
		}
	})

	if err != nil {
		_ = sse.ConsoleError(fmt.Errorf("ChatStream error: %w", err))
		return
	}
}

// Start is a function that starts the Orus API server
// It sets up the routes and starts the server
func (s *OrusAPI) Start() {
	s.setupRoutes()
	log.Println("Orus API ORUS_API_PORT", LoadEnv("ORUS_API_PORT"))
	log.Println("Orus API ORUS_API_AGENT_MEMORY_PATH", LoadEnv("ORUS_API_AGENT_MEMORY_PATH"))
	log.Println("Orus API ORUS_API_TOK_PATH", LoadEnv("ORUS_API_TOK_PATH"))
	log.Println("Orus API ORUS_API_ONNX_PATH", LoadEnv("ORUS_API_ONNX_PATH"))
	log.Println("Orus API ORUS_API_ONNX_RUNTIME_PATH", LoadEnv("ORUS_API_ONNX_RUNTIME_PATH"))
	log.Println("Orus API ORUS_API_OLLAMA_BASE_URL", LoadEnv("ORUS_API_OLLAMA_BASE_URL"))
	log.Println("Orus API server started on port", s.server.Addr)

	if err := s.server.ListenAndServe(); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

// SetVerbose is a function that sets the verbose mode
// It returns the OrusAPI instance
func (s *OrusAPI) SetVerbose(verbose bool) *OrusAPI {
	s.Verbose = verbose
	return s
}

// GetUsers godoc
// @Summary      Returns the system information
// @Description  Returns the system information
// @Tags         system
// @Accept       json
// @Produce      json
// @Success      200  {object}  OrusResponse
// @Failure      500  {object}  OrusResponse
// @Router       /orus-api/v1/system-info [get]
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

// EmbedText godoc
// @Summary      Embeds text using the BGE-M3 or Ollama embedding model
// @Description  Embeds text using the BGE-M3 or Ollama embedding model
// @Tags         embed
// @Accept       json
// @Produce      json
// @Success      200  {object}  OrusResponse
// @Failure      500  {object}  OrusResponse
// @Router       /orus-api/v1/embed-text [post]
func (s *OrusAPI) EmbedText(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()

	type Req struct {
		Model string `json:"model"`
		Text  string `json:"text"`
	}

	request := new(Req)

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		respondError(w, http.StatusBadRequest, "invalid_request", "Invalid JSON body: "+err.Error())
		return
	}

	model := request.Model
	if model == "" {
		respondError(w, http.StatusBadRequest, "missing_model", "Field 'model' is required")
		return
	}

	text := request.Text
	if text == "" {
		respondError(w, http.StatusBadRequest, "missing_text", "Field 'text' is required")
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
	defer cancel()

	respChan := make(chan *OrusResponse, 1)

	go func() {
		resp := s.embedText(model, text, startTime)
		select {
		case respChan <- resp:
		case <-ctx.Done():
		}
	}()

	select {
	case resp := <-respChan:
		respondJSON(w, http.StatusOK, resp)
	case <-ctx.Done():
		timeoutResp := NewOrusResponse()
		timeoutResp.Error = "Error Timeout"
		timeoutResp.Success = false
		timeoutResp.TimeTaken = time.Since(startTime)
		timeoutResp.Message = "Error Timeout"
		respondJSON(w, http.StatusGatewayTimeout, timeoutResp)
	}
}

// OllamaModelList godoc
// @Summary      Returns the list of models available in the Ollama server
// @Description  Returns the list of models available in the Ollama server
// @Tags         ollama
// @Accept       json
// @Produce      json
// @Success      200  {object}  OrusResponse
// @Failure      500  {object}  OrusResponse
// @Router       /orus-api/v1/ollama-model-list [get]
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

// OllamaPullModel godoc
// @Summary      Pulls a model from the Ollama server
// @Description  Pulls a model from the Ollama server
// @Tags         ollama
// @Accept       json
// @Produce      json
// @Success      200  {object}  OrusResponse
// @Failure      500  {object}  OrusResponse
// @Router       /orus-api/v1/ollama-pull-model [post]
func (s *OrusAPI) OllamaPullModel(w http.ResponseWriter, r *http.Request) {
	type Req struct {
		Name string `json:"name"`
	}

	request := new(Req)

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		respondError(w, http.StatusBadRequest, "invalid_request", "Invalid JSON body: "+err.Error())
		return
	}

	if request.Name == "" {
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

	ctx := r.Context()

	progressCallback := func(progress PullModelProgress) {
		select {
		case <-ctx.Done():
			return
		default:
			data, _ := json.Marshal(progress)
			if _, err := fmt.Fprintf(w, "data: %s\n\n", string(data)); err != nil {
				return
			}
			flusher.Flush()
		}
	}

	if err := s.OllamaClient.PullModel(request.Name, progressCallback); err != nil {
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
		"message": fmt.Sprintf("Model %s downloaded successfully", request.Name),
	})
	fmt.Fprintf(w, "data: %s\n\n", string(successData))
	flusher.Flush()
}

func (s *OrusAPI) embedText(model string, text string, startTime time.Time) *OrusResponse {
	resp := NewOrusResponse()

	var (
		serial       string
		vector       []any
		dimensions   int
		quantization string
	)
	serial = uuid.New().String()
	switch model {
	case "bge-m3":
		vector32, err := s.Orus.BGEM3Embedder.Embed(text)
		if err != nil {
			resp.Error = err.Error()
			resp.Success = false
			resp.TimeTaken = time.Since(startTime)
			resp.Message = fmt.Sprintf("Error embedding text with model %s", model)
			return resp
		}
		vector = make([]any, len(vector32))
		for i, v := range vector32 {
			vector[i] = v
		}
		dimensions = len(vector32)
		quantization = "float32"
	case "nomic-embed-text:latest":
		vector64, err := s.Orus.OllamaClient.GetEmbedding(model, text)
		if err != nil {
			resp.Error = err.Error()
			resp.Success = false
			resp.TimeTaken = time.Since(startTime)
			resp.Message = fmt.Sprintf("Error embedding text with model %s", model)
			return resp
		}
		vector = make([]any, len(vector64))
		for i, v := range vector64 {
			vector[i] = v
		}
		dimensions = len(vector64)
		quantization = "float64"
	case "ollama-bge-m3":
		vector64, err := s.Orus.OllamaClient.GetEmbedding("bge-m3:latest", text)
		if err != nil {
			resp.Error = err.Error()
			resp.Success = false
			resp.TimeTaken = time.Since(startTime)
			resp.Message = fmt.Sprintf("Error embedding text with model %s", model)
			return resp
		}
		vector = make([]any, len(vector64))
		for i, v := range vector64 {
			vector[i] = v
		}
		dimensions = len(vector64)
		quantization = "float64"
	default:
		resp.Error = "Invalid model"
		resp.Success = false
		resp.TimeTaken = time.Since(startTime)
		resp.Message = "Invalid model"
		return resp
	}
	resp.Data = map[string]interface{}{
		"serial":       serial,
		"vector":       vector,
		"text":         text,
		"model":        model,
		"dimensions":   dimensions,
		"quantization": quantization,
	}
	resp.Success = true
	resp.TimeTaken = time.Since(startTime)
	resp.Message = "Embed request received successfully"
	return resp
}

// CallLLM godoc
// @Summary      Calls the LLM using the Ollama client
// @Description  Calls the LLM using the Ollama client
// @Tags         llm
// @Accept       json
// @Produce      json
// @Success      200  {object}  OrusResponse
// @Failure      500  {object}  OrusResponse
// @Router       /orus-api/v1/call-llm [post]
func (s *OrusAPI) CallLLM(w http.ResponseWriter, r *http.Request) {

	startTime := time.Now()

	response := NewOrusResponse()
	request := new(OrusRequest)

	request.Body = make(map[string]interface{})
	if err := json.NewDecoder(r.Body).Decode(&request.Body); err != nil {
		respondError(w, http.StatusBadRequest, "invalid_request", "Invalid JSON body: "+err.Error())
		return
	}

	data := request.Body["body"].(map[string]interface{})

	modelVal, ok := data["model"]
	if !ok {
		respondError(w, http.StatusBadRequest, "missing_model", "Field 'model' is required")
		return
	}
	model, ok := modelVal.(string)
	if !ok {
		respondError(w, http.StatusBadRequest, "invalid_model", "Field 'model' must be a string")
		return
	}

	thinkValVal, ok := data["think"]
	if !ok {
		respondError(w, http.StatusBadRequest, "missing_think", "Field 'think' is required")
		return
	}
	think, ok := thinkValVal.(bool)
	if !ok {
		respondError(w, http.StatusBadRequest, "invalid_think", "Field 'think' must be a boolean")
		return
	}

	messagesRaw, ok := data["messages"]
	if !ok {
		respondError(w, http.StatusBadRequest, "missing_messages", "Field 'messages' is required")
		return
	}

	messagesJSON, err := json.Marshal(messagesRaw)
	if err != nil {
		respondError(w, http.StatusBadRequest, "invalid_messages", "Error marshalling messages")
		return
	}

	var messages []Message
	if err := json.Unmarshal(messagesJSON, &messages); err != nil {
		respondError(w, http.StatusBadRequest, "invalid_messages", "Error unmarshalling messages: "+err.Error())
		return
	}

	stream := false
	if val, ok := data["stream"]; ok {
		if b, ok := val.(bool); ok {
			stream = b
		}
	}

	chatRequest := ChatRequest{
		Model:    model,
		Messages: messages,
		Stream:   stream,
		Think:    think,
	}

	formatValVal, ok := data["format"]
	if ok {
		format, _ := formatValVal.(string)
		chatRequest.Format = format
	}

	if imagesVal, ok := data["images"].([]interface{}); ok {
		chatRequest.Images = ConvertInterfaceToStrings(imagesVal)
	}

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
		err := s.OllamaClient.ChatStream(chatRequest, chatStreamProgressCallback)
		if err != nil {
			errorData, _ := json.Marshal(map[string]string{
				"status": "error",
				"error":  err.Error(),
			})
			fmt.Fprintf(w, "data: %s\n\n", string(errorData))
			flusher.Flush()
			return
		}
		successData, _ := json.Marshal(map[string]interface{}{
			"status":     "success",
			"message":    "LLM request received successfully",
			"content":    strings.Join(content, ""),
			"serial":     uuid.New().String(),
			"time_taken": time.Since(startTime).String(),
			"model":      model,
			"stream":     true,
		})
		fmt.Fprintf(w, "data: %s\n\n", string(successData))
		flusher.Flush()
		return
	} else {
		responseLLM, err := s.OllamaClient.Chat(chatRequest)
		if err != nil {
			response.Error = err.Error()
			response.Message = "Error calling LLM"
			response.Success = false
			response.TimeTaken = time.Since(startTime)
			respondJSON(w, http.StatusInternalServerError, response)
		} else {
			successData := map[string]interface{}{
				"success":    true,
				"message":    "LLM request received successfully",
				"content":    responseLLM.Message.Content,
				"serial":     uuid.New().String(),
				"time_taken": time.Since(startTime).String(),
				"model":      model,
				"stream":     stream,
				"think":      think,
			}
			respondJSON(w, http.StatusOK, successData)
		}
	}
}

// CallLLM godoc
// @Summary      Calls the LLM using the Ollama client
// @Description  Calls the LLM using the Ollama client
// @Tags         llm
// @Accept       json
// @Produce      json
// @Success      200  {object}  OrusResponse
// @Failure      500  {object}  OrusResponse
// @Router       /orus-api/v1/call-llm [post]
func (s *OrusAPI) CallLLMCloud(w http.ResponseWriter, r *http.Request) {

	startTime := time.Now()

	response := NewOrusResponse()
	request := new(OrusRequest)

	request.Body = make(map[string]interface{})
	if err := json.NewDecoder(r.Body).Decode(&request.Body); err != nil {
		respondError(w, http.StatusBadRequest, "invalid_request", "Invalid JSON body: "+err.Error())
		return
	}

	data := request.Body["body"].(map[string]interface{})

	modelVal, ok := data["model"]
	if !ok {
		respondError(w, http.StatusBadRequest, "missing_model", "Field 'model' is required")
		return
	}
	model, ok := modelVal.(string)
	if !ok {
		respondError(w, http.StatusBadRequest, "invalid_model", "Field 'model' must be a string")
		return
	}

	thinkValVal, ok := data["think"]
	if !ok {
		respondError(w, http.StatusBadRequest, "missing_think", "Field 'think' is required")
		return
	}
	think, ok := thinkValVal.(bool)
	if !ok {
		respondError(w, http.StatusBadRequest, "invalid_think", "Field 'think' must be a boolean")
		return
	}

	messagesRaw, ok := data["messages"]
	if !ok {
		respondError(w, http.StatusBadRequest, "missing_messages", "Field 'messages' is required")
		return
	}

	messagesJSON, err := json.Marshal(messagesRaw)
	if err != nil {
		respondError(w, http.StatusBadRequest, "invalid_messages", "Error marshalling messages")
		return
	}

	var messages []Message
	if err := json.Unmarshal(messagesJSON, &messages); err != nil {
		respondError(w, http.StatusBadRequest, "invalid_messages", "Error unmarshalling messages: "+err.Error())
		return
	}

	stream := false
	if val, ok := data["stream"]; ok {
		if b, ok := val.(bool); ok {
			stream = b
		}
	}

	chatRequest := ChatRequest{
		Model:    model,
		Messages: messages,
		Stream:   stream,
		Think:    think,
	}

	formatValVal, ok := data["format"]
	if ok {
		format, _ := formatValVal.(string)
		chatRequest.Format = format
	}

	if imagesVal, ok := data["images"].([]interface{}); ok {
		chatRequest.Images = ConvertInterfaceToStrings(imagesVal)
	}

	chatRequest.Model = model

	log.Println("chatRequest--->", chatRequest)
	log.Println("model--->", model)
	log.Println("think--->", think)
	log.Println("stream--->", stream)

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
		err := s.OllamaClient.ChatStreamCloud(chatRequest, chatStreamProgressCallback)
		if err != nil {
			errorData, _ := json.Marshal(map[string]string{
				"status": "error",
				"error":  err.Error(),
			})
			fmt.Fprintf(w, "data: %s\n\n", string(errorData))
			flusher.Flush()
			return
		}
		successData, _ := json.Marshal(map[string]interface{}{
			"status":     "success",
			"message":    "LLM request received successfully",
			"content":    strings.Join(content, ""),
			"serial":     uuid.New().String(),
			"time_taken": time.Since(startTime).String(),
			"model":      model,
			"stream":     true,
			"think":      think,
		})
		fmt.Fprintf(w, "data: %s\n\n", string(successData))
		flusher.Flush()
		return
	} else {
		responseLLM, err := s.OllamaClient.ChatCloud(chatRequest)
		if err != nil {
			response.Error = err.Error()
			response.Message = "Error calling LLM"
			response.Success = false
			response.TimeTaken = time.Since(startTime)
			respondJSON(w, http.StatusInternalServerError, response)
		} else {
			successData := map[string]interface{}{
				"success":    true,
				"message":    "LLM request received successfully",
				"content":    responseLLM.Message.Content,
				"serial":     uuid.New().String(),
				"time_taken": time.Since(startTime).String(),
				"model":      model,
				"stream":     stream,
				"think":      think,
			}
			respondJSON(w, http.StatusOK, successData)
		}
	}
}

func (s *OrusAPI) handleStreamingResponseChi(ctx context.Context, w http.ResponseWriter, chatRequest *ChatRequest, startTime time.Time, requestID string) {
	// Headers SSE
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.Header().Set("X-Request-ID", requestID)

	flusher, ok := w.(http.Flusher)
	if !ok {
		respondError(w, http.StatusInternalServerError, "streaming_not_supported", "Streaming not supported")
		return
	}

	// StringBuilder do pool
	contentBuilder := stringBuilderPool.Get().(*strings.Builder)
	contentBuilder.Reset()
	defer stringBuilderPool.Put(contentBuilder)

	// Buffer para JSON
	jsonBuf := bufferPool.Get().(*bytes.Buffer)
	defer bufferPool.Put(jsonBuf)

	encoder := json.NewEncoder(jsonBuf)
	flusher.Flush()

	// Canal para erros do streaming
	errChan := make(chan error, 1)

	// Callback do streaming
	chatStreamProgressCallback := func(chatResp ChatStreamResponse) {
		select {
		case <-ctx.Done():
			return
		default:
			jsonBuf.Reset()
			if err := encoder.Encode(chatResp); err != nil {
				return
			}
			data := bytes.TrimSuffix(jsonBuf.Bytes(), []byte("\n"))
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
			contentBuilder.WriteString(chatResp.Message.Content)
		}
	}

	// Executar streaming em goroutine para permitir cancelamento
	go func() {
		errChan <- s.OllamaClient.ChatStreamCloud(*chatRequest, chatStreamProgressCallback)
	}()

	// Aguardar resultado ou cancelamento
	select {
	case <-ctx.Done():
		jsonBuf.Reset()
		encoder.Encode(map[string]string{
			"status": "cancelled",
			"error":  "Request cancelled by client",
		})
		fmt.Fprintf(w, "data: %s\n\n", jsonBuf.String())
		flusher.Flush()
		return

	case err := <-errChan:
		if err != nil {
			jsonBuf.Reset()
			encoder.Encode(map[string]string{
				"status": "error",
				"error":  err.Error(),
			})
			fmt.Fprintf(w, "data: %s\n\n", jsonBuf.String())
			flusher.Flush()
			return
		}
	}

	jsonBuf.Reset()
	encoder.Encode(map[string]interface{}{
		"status":     "success",
		"message":    "LLM request completed successfully",
		"content":    contentBuilder.String(),
		"serial":     uuid.New().String(),
		"request_id": requestID,
		"time_taken": time.Since(startTime).String(),
		"model":      chatRequest.Model,
		"stream":     true,
		"think":      chatRequest.Think,
	})
	fmt.Fprintf(w, "data: %s\n\n", jsonBuf.String())
	flusher.Flush()
}

func (s *OrusAPI) CallLLMOptimized(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()
	ctx := r.Context()

	requestID := middleware.GetReqID(ctx)

	buf := bufferPool.Get().(*bytes.Buffer)
	buf.Reset()
	defer bufferPool.Put(buf)

	if _, err := io.Copy(buf, r.Body); err != nil {
		respondError(w, http.StatusBadRequest, "read_error", "Failed to read request body")
		return
	}

	var request LLMCloudRequest
	if err := json.Unmarshal(buf.Bytes(), &request); err != nil {
		respondError(w, http.StatusBadRequest, "invalid_request", "Invalid JSON body: "+err.Error())
		return
	}

	if err := validateLLMRequest(&request.Body); err != nil {
		respondError(w, http.StatusBadRequest, err.Code, err.Message)
		return
	}

	chatRequest := acquireChatRequest(&request.Body)
	defer releaseChatRequest(chatRequest)

	go logRequest(requestID, chatRequest)

	if chatRequest.Stream {
		s.handleStreamingResponseChi(ctx, w, chatRequest, startTime, requestID)
	} else {
		s.handleSyncResponseChi(ctx, w, chatRequest, startTime, requestID)
	}
}

func (s *OrusAPI) HealthCheck(w http.ResponseWriter, r *http.Request) {
	respondJSON(w, http.StatusOK, map[string]interface{}{
		"status": "healthy",
		"time":   time.Now().UTC(),
	})
}

func (s *OrusAPI) handleSyncResponseChi(ctx context.Context, w http.ResponseWriter, chatRequest *ChatRequest, startTime time.Time, requestID string) {

	type result struct {
		response *ChatResponse
		err      error
	}
	resultChan := make(chan result, 1)

	go func() {
		resp, err := s.OllamaClient.ChatCloud(*chatRequest)
		resultChan <- result{resp, err}
	}()

	select {
	case <-ctx.Done():
		respondError(w, http.StatusRequestTimeout, "timeout", "Request timed out or was cancelled")
		return

	case res := <-resultChan:
		if res.err != nil {
			response := OrusResponse{
				Success:   false,
				Serial:    requestID,
				Error:     res.err.Error(),
				Message:   "Error calling LLM",
				TimeTaken: time.Since(startTime),
			}
			respondJSON(w, http.StatusInternalServerError, response)
			return
		}

		successData := map[string]interface{}{
			"success":    true,
			"message":    "LLM request completed successfully",
			"content":    res.response.Message.Content,
			"serial":     uuid.New().String(),
			"request_id": requestID,
			"time_taken": time.Since(startTime).String(),
			"model":      chatRequest.Model,
			"stream":     false,
			"think":      chatRequest.Think,
		}
		respondJSON(w, http.StatusOK, successData)
	}
}

func (s *OrusAPI) HandleOpenRouteChatStream(w http.ResponseWriter, r *http.Request) {
	requestID := r.Context().Value("requestID").(string)

	// Parse do request
	var req OpenRouteChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.jsonError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	if len(req.Messages) == 0 {
		respondError(w, http.StatusBadRequest, "messages required", "Messages required")
		return
	}

	// Converte para DTO do OpenRoute
	openRouteReq := &OpenRoute{
		Model:       req.Model,
		Models:      req.Models,
		Stream:      true,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		Messages:    make([]OpenRouteMessagesDTO, len(req.Messages)),
	}

	for i, msg := range req.Messages {
		openRouteReq.Messages[i] = OpenRouteMessagesDTO{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Minute)

	// Registra conexão ativa
	s.activeConnsMu.Lock()
	s.activeConns[requestID] = cancel
	s.activeConnsMu.Unlock()

	defer func() {
		cancel()
		s.activeConnsMu.Lock()
		delete(s.activeConns, requestID)
		s.activeConnsMu.Unlock()
	}()

	// Inicia streaming
	openRouteClient := NewOpenRouteWrapper()
	eventChan, err := openRouteClient.CallOpenRouterAPIStream(ctx, openRouteReq)
	if err != nil {
		respondError(w, http.StatusBadGateway, "failed to connect to AI provider", "Failed to connect to AI provider")
		return
	}

	// Configura headers SSE
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no") // Desabilita buffering do nginx
	w.Header().Set("Access-Control-Allow-Origin", "*")

	flusher, ok := w.(http.Flusher)
	if !ok {
		respondError(w, http.StatusInternalServerError, "streaming not supported", "Streaming not supported")
		return
	}

	// Processa eventos do stream
	for event := range eventChan {
		select {
		case <-ctx.Done():
			s.writeSSE(w, flusher, "error", `{"error":"connection closed"}`)
			return
		default:
		}

		if event.Error != nil {
			respondError(w, http.StatusInternalServerError, "stream error", "Stream error: "+event.Error.Error())
			s.writeSSE(w, flusher, "error", fmt.Sprintf(`{"error":"%s"}`, event.Error.Error()))
			return
		}

		if event.Done {
			s.writeSSE(w, flusher, "done", `{"status":"complete"}`)
			return
		}

		if event.Chunk != nil {
			chunkJSON, err := json.Marshal(event.Chunk)
			if err != nil {
				continue
			}
			s.writeSSE(w, flusher, "message", string(chunkJSON))
		}
	}
}

func (s *OrusAPI) HandleOpenRouteChatCredit(w http.ResponseWriter, r *http.Request) {
	openRouteClient := NewOpenRouteWrapper()
	credits := openRouteClient.GetCurrentCredits()
	respondJSON(w, http.StatusOK, credits)
}

func (s *OrusAPI) HandleWebSearch(w http.ResponseWriter, r *http.Request) {
	var req WebSearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.jsonError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	ollamaClient := NewOllamaClient()
	results, err := ollamaClient.WebSearch(req.Query, req.MaxResults)
	if err != nil {
		respondError(w, http.StatusInternalServerError, "failed to search web", "Failed to search web")
		return
	}
	respondJSON(w, http.StatusOK, results)
}

// ==================== Helper Functions ====================

func acquireChatRequest(body *LLMCloudRequestBody) *ChatRequest {
	chatRequest := chatRequestPool.Get().(*ChatRequest)
	chatRequest.Model = body.Model
	chatRequest.Messages = append(chatRequest.Messages[:0], body.Messages...)
	chatRequest.Stream = body.Stream
	chatRequest.Think = body.Think
	chatRequest.Format = body.Format
	if len(body.Images) > 0 {
		chatRequest.Images = append(chatRequest.Images[:0], body.Images...)
	}
	return chatRequest
}

func (s *OrusAPI) CancelConnection(requestID string) bool {
	s.activeConnsMu.Lock()
	defer s.activeConnsMu.Unlock()

	if cancel, exists := s.activeConns[requestID]; exists {
		cancel()
		delete(s.activeConns, requestID)
		return true
	}
	return false
}

// CancelAllConnections cancela todas as conexões ativas
func (s *OrusAPI) CancelAllConnections() int {
	s.activeConnsMu.Lock()
	defer s.activeConnsMu.Unlock()

	count := len(s.activeConns)
	for id, cancel := range s.activeConns {
		cancel()
		delete(s.activeConns, id)
	}
	return count
}

func (s *OrusAPI) writeSSE(w http.ResponseWriter, flusher http.Flusher, event, data string) {
	fmt.Fprintf(w, "event: %s\n", event)
	fmt.Fprintf(w, "data: %s\n\n", data)
	flusher.Flush()
}

func (s *OrusAPI) jsonError(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]string{"error": message})
}

// ---------------------------MAIN FUNCTION------------------------------

func main() {
	orusApi := NewOrusAPI()
	orusApi.Start()
}
