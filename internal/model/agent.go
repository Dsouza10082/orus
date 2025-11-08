package model

import "time"

type Agent struct {
	Serial    string                 `json:"serial" db:"serial"`
	Name      string                 `json:"name" db:"name"`
	Created   string                 `json:"created" db:"created"`
	Updated   string                 `json:"updated" db:"updated"`
	Status    string                 `json:"status" db:"status"`
	Message   string                 `json:"message" db:"message"`
	Data      map[string]interface{} `json:"data" db:"data"`
	Error     string                 `json:"error" db:"error"`
	TimeTaken time.Duration          `json:"time_taken" db:"time_taken"`
}
