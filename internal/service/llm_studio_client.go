package service

import (
	"log"
	"github.com/nats-io/nats.go"
	"github.com/Dsouza10082/nats-llm-studio"
	"github.com/Dsouza10082/orus/config"
)

func NewNATSClient() *nats_llm_studio.Server {
	params := config.GetParameters()
	nc, err := nats.Connect(params.NatsURL)
	if err != nil {
		log.Fatal(err)
	}
	lmStudioClient := nats_llm_studio.NewLMStudioClient(params.LMStudioBaseURL, params.NatsModelsDir)
	return nats_llm_studio.NewServer(lmStudioClient, nc)
}