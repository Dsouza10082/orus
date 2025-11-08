package config

import (
	"log"
	"os"
	"strconv"

	"github.com/joho/godotenv"
)

type Parameters struct {
	OrusAPIPort                string `json:"orus_api_port"`
	OllamaBaseURL              string `json:"ollama_base_url"`
	OnnxPath                   string `json:"onnx_path"`
	TokPath                    string `json:"tok_path"`
	CachePort                  string `json:"cache_port"`
	CachePassword              string `json:"cache_password"`
	AgentMemoryPath            string `json:"agent_memory_path"`
	OnnxRuntimePath            string `json:"onnx_runtime_path"`
	MilvusHost                 string `json:"milvus_host"`
	MilvusUser                 string `json:"milvus_user"`
	MilvusPassword             string `json:"milvus_password"`
	MilvusDatabase             string `json:"milvus_database"`
	MilvusMaxConnections       int    `json:"milvus_max_connections"`
	MilvusMinConnections       int    `json:"milvus_min_connections"`
	MilvusConnectionLifeTime   int    `json:"milvus_connection_life_time"`
	PostgresHost               string `json:"postgres_host"`
	PostgresUser               string `json:"postgres_user"`
	PostgresPassword           string `json:"postgres_password"`
	PostgresDatabase           string `json:"postgres_database"`
	PostgresMaxConnections     int    `json:"postgres_max_connections"`
	PostgresMinConnections     int    `json:"postgres_min_connections"`
	PostgresConnectionLifeTime int    `json:"postgres_connection_life_time"`
}

func GetParameters() *Parameters {
	pgMaxConnectionsAux := LoadEnv("PG_MAX_CONNECTIONS")
	pgMinConnectionsAux := LoadEnv("PG_MIN_CONNECTIONS")
	pgConnectionLifeTimeAux := LoadEnv("PG_CONNECTION_LIFE_TIME")
	milvusMaxConnectionsAux := LoadEnv("MILVUS_MAX_CONNECTIONS")
	milvusMinConnectionsAux := LoadEnv("MILVUS_MIN_CONNECTIONS")
	milvusConnectionLifeTimeAux := LoadEnv("MILVUS_CONNECTION_LIFE_TIME")
	
	pgMaxConnections, err := strconv.Atoi(pgMaxConnectionsAux)
	if err != nil {
		panic(err)
	}
	pgMinConnections, err := strconv.Atoi(pgMinConnectionsAux)
	if err != nil {
		panic(err)
	}
	pgConnectionLifeTime, err := strconv.Atoi(pgConnectionLifeTimeAux)
	if err != nil {
		panic(err)
	}
	milvusMaxConnections, err := strconv.Atoi(milvusMaxConnectionsAux)
	if err != nil {
		panic(err)
	}
	milvusMinConnections, err := strconv.Atoi(milvusMinConnectionsAux)
	if err != nil {
		panic(err)
	}
	milvusConnectionLifeTime, err := strconv.Atoi(milvusConnectionLifeTimeAux)
	if err != nil {
		panic(err)
	}
	return &Parameters{
		OrusAPIPort:                LoadEnv("ORUS_API_PORT"),
		OllamaBaseURL:              LoadEnv("ORUS_API_OLLAMA_BASE_URL"),
		AgentMemoryPath:            LoadEnv("ORUS_API_AGENT_MEMORY_PATH"),
		OnnxPath:                   LoadEnv("ORUS_API_ONNX_PATH"),
		TokPath:                    LoadEnv("ORUS_API_TOK_PATH"),
		OnnxRuntimePath:            LoadEnv("ORUS_API_ONNX_RUNTIME_PATH"),
		MilvusHost:                 LoadEnv("MILVUS_HOST"),
		MilvusUser:                 LoadEnv("MILVUS_USER"),
		MilvusPassword:             LoadEnv("MILVUS_PASSWORD"),
		MilvusDatabase:             LoadEnv("MILVUS_DATABASE"),
		MilvusMaxConnections:       milvusMaxConnections,
		MilvusMinConnections:       milvusMinConnections,
		MilvusConnectionLifeTime:   milvusConnectionLifeTime,
		PostgresHost:               LoadEnv("PG_HOST"),
		PostgresUser:               LoadEnv("PG_USER"),
		PostgresPassword:           LoadEnv("PG_PASSWORD"),
		PostgresDatabase:           LoadEnv("PG_DATABASE"),
		PostgresMaxConnections:     pgMaxConnections,
		PostgresMinConnections:     pgMinConnections,
		PostgresConnectionLifeTime: pgConnectionLifeTime,
	}
}

func LoadEnv(key string) string {
	err := godotenv.Load("../../.env")
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
