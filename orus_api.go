package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
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
	Port    string
	router  *chi.Mux
	Verbose bool
}

type PromptSignals struct {
    Prompt        string `json:"prompt"`
    Model         string `json:"model"`
    OperationType string `json:"operationType"`
    ResponseMode  string `json:"responseMode"`
    Result        string `json:"result"`
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
	s.router.Get("/prompt", s.IndexHandler)
	s.router.Post("/prompt/llm-stream", s.PromptLLMStream)

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
	indexView.RenderIndex(w)
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

	if signals.Model == "" {
		signals.Model = "llama-3"
	}
	if signals.ResponseMode == "" {
		signals.ResponseMode = "stream"
	}

	sse := datastar.NewSSE(w, r)
	defer func() {
		_ = sse.ConsoleLog("PromptLLMStream closed")
	}()

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
	log.Println("Orus API ORUS_API_AGENT_MEMORY_PATH", LoadEnv("ORUS_API_AGENT_MEMORY_PATH"))
	log.Println("Orus API ORUS_API_TOK_PATH", LoadEnv("ORUS_API_TOK_PATH"))
	log.Println("Orus API ORUS_API_ONNX_PATH", LoadEnv("ORUS_API_ONNX_PATH"))
	log.Println("Orus API ORUS_API_ONNX_RUNTIME_PATH", LoadEnv("ORUS_API_ONNX_RUNTIME_PATH"))
	log.Println("Orus API ORUS_API_OLLAMA_BASE_URL", LoadEnv("ORUS_API_OLLAMA_BASE_URL"))

	log.Printf("Orus API server started on port %s", s.Port)
	if err := http.ListenAndServe(":"+s.Port, s.router); err != nil {
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

// decodeJSONBody is a function that decodes the JSON body of the request
// It returns true if the body is decoded successfully, false otherwise
func decodeJSONBody(w http.ResponseWriter, r *http.Request, dst any, maxBytes int64) bool {
	defer r.Body.Close()

	if maxBytes > 0 {
		r.Body = http.MaxBytesReader(w, r.Body, maxBytes)
	}

	if err := json.NewDecoder(r.Body).Decode(dst); err != nil {
		respondError(w, http.StatusBadRequest, "invalid_request", "Invalid JSON body: "+err.Error())
		return false
	}
	return true
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

	request := new(OrusRequest)
	if !decodeJSONBody(w, r, request, 1<<20) {
		return
	}

	modelVal, ok := request.Body["model"]
	if !ok {
		respondError(w, http.StatusBadRequest, "missing_model", "Field 'model' is required")
		return
	}
	model, ok := modelVal.(string)
	if !ok {
		respondError(w, http.StatusBadRequest, "invalid_model", "Field 'model' must be a string")
		return
	}

	textVal, ok := request.Body["text"]
	if !ok {
		respondError(w, http.StatusBadRequest, "missing_text", "Field 'text' is required")
		return
	}
	text, ok := textVal.(string)
	if !ok {
		respondError(w, http.StatusBadRequest, "invalid_text", "Field 'text' must be a string")
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
	var req struct {
		Name string `json:"name"`
	}

	if !decodeJSONBody(w, r, &req, 1<<20) {
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

	if err := s.OllamaClient.PullModel(req.Name, progressCallback); err != nil {
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
	case "nomic-embed-text":
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

	if !decodeJSONBody(w, r, request, 2<<20) {
		return
	}

	modelVal, ok := request.Body["model"]
	if !ok {
		respondError(w, http.StatusBadRequest, "missing_model", "Field 'model' is required")
		return
	}
	model, ok := modelVal.(string)
	if !ok {
		respondError(w, http.StatusBadRequest, "invalid_model", "Field 'model' must be a string")
		return
	}

	messagesRaw, ok := request.Body["messages"]
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
	if val, ok := request.Body["stream"]; ok {
		if b, ok := val.(bool); ok {
			stream = b
		}
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
		err := s.OllamaClient.ChatStream(ChatRequest{
			Model:    model,
			Messages: messages,
			Stream:   stream,
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
		responseLLM, err := s.OllamaClient.Chat(ChatRequest{
			Model:    model,
			Messages: messages,
			Stream:   stream,
		})
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
			}
			respondJSON(w, http.StatusOK, successData)
		}
	}
}

// ---------------------------MAIN FUNCTION------------------------------

func main() {
	orusApi := NewOrusAPI()
	orusApi.Start()
}
