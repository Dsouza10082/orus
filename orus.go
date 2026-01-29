package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"

	bge_m3 "github.com/Dsouza10082/go-bge-m3-embed"
	"github.com/joho/godotenv"
)

type Orus struct {
	BGEM3Embedder *bge_m3.GolangBGE3M3Embedder
	OrusAPI       *OrusAPI
	OllamaClient  *OllamaClient
}

func NewOrus() *Orus {
	bge_m3_embedder := bge_m3.NewGolangBGE3M3Embedder().
		SetMemoryPath(LoadEnv("ORUS_API_AGENT_MEMORY_PATH")).
		SetTokPath(LoadEnv("ORUS_API_TOK_PATH")).
		SetOnnxPath(LoadEnv("ORUS_API_ONNX_PATH")).
		SetRuntimePath(LoadEnv("ORUS_API_ONNX_RUNTIME_PATH"))
	bge_m3_embedder.EmbeddingModel.SetOnnxModelPath(LoadEnv("ORUS_API_ONNX_PATH"))
	bge_m3_embedder.Verbose = true
	ollamaClient := NewOllamaClient()
	return &Orus{
		BGEM3Embedder: bge_m3_embedder,
		OllamaClient: ollamaClient,
	}
}

func (s *Orus) EmbedWithBGE_M3(text string) ([]float32, error) {
	vector, err := s.BGEM3Embedder.Embed(text)
	if err != nil {
		log.Println("Error embedding text: ", err)
		return nil, err
	}
	return vector, nil
}

func (s *Orus) EmbedWithOllama(text string) ([]float64, error) {
	vector, err := s.OllamaClient.GetEmbedding("nomic-embed-text", text)
	if err != nil {
		log.Println("Error embedding text: ", err)
		return nil, err
	}
	return vector, nil
}

func (s *Orus) CallLLM(model string, messages []Message, stream bool) (string, error) {
	response, err := s.OllamaClient.Chat(ChatRequest{
		Model: model,
		Messages: messages,
		Stream: stream,
	})
	if err != nil {
		log.Println("Error chatting: ", err)
		return "", err
	}
	return response.Message.Content, nil
}

func (s *Orus) PullLLMModel(model string) (string, error) {

	progressCallback := func(progress PullModelProgress) {
		data, _ := json.Marshal(progress)
		fmt.Println(string(data))
	}

	err := s.OllamaClient.PullModel(model, progressCallback)
	if err != nil {
		return "Error pulling model: " + err.Error(), err
	}

	return "Model pulled successfully", nil
}

func LoadEnv(key string) string {
	err := godotenv.Load(".env")
	if err != nil {
		log.Println("Error loading.env file " + err.Error())
		return ""
	}
	value := os.Getenv(key)
	if value == "" {
		log.Println("Environment variable " + key + " is not set")
		return ""
	}
	return value
}
