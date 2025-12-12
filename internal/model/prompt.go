package model

type Prompt struct {
	SerialModel string `json:"serial_model"`
	Prompt      string `json:"prompt"`
	Description string `json:"description"`
	Created     string `json:"created"`
	Updated     string `json:"updated"`
	Status      string `json:"status"`
	Rating      int    `json:"rating"`
}
