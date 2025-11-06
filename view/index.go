package view

import (
	_ "embed"
	"net/http"
)

//go:embed index.html
var indexHTML string

type View struct {
   
}

func NewView() *View {
	return &View{
		
	}
}

func (v *View) RenderIndex(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/html")
	w.Write([]byte(indexHTML))
}