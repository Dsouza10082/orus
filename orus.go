package orus

import (
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
		SetMemoryPath(LoadEnv("AGENT_MEMORY_PATH")).
		SetTokPath(LoadEnv("TOK_PATH")).
		SetOnnxPath(LoadEnv("ONNX_PATH")).
		SetRuntimePath(LoadEnv("ONNX_RUNTIME_PATH"))

	ollamaClient := NewOllamaClient(LoadEnv("OLLAMA_BASE_URL"))

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

func LoadEnv(key string) string {
	env := os.Getenv("ENV_TYPE")
	if env != "" {
		return os.Getenv(key)
	}
	err := godotenv.Load(".env")
	if err != nil {
		log.Println("Error loading.env file " + err.Error())
		os.Exit(1)
	}
	return os.Getenv(key)
}
