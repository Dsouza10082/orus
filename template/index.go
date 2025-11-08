package template

import (
	_ "embed"
	"fmt"
	"net/http"
)

//go:embed index.html
var indexHTML string

type View struct {
	models      []string
	optionsHtml string
}

func NewView() *View {
	return &View{
		models: []string{},
	}
}

func (v *View) SetModels(models []string) *View {
	v.models = models
	return v
}

func (v *View) renderOptions() *View {
	optionsHtml := ""
	optionsHtml += "<option class='bg-white' value=''>Select a model</option>"
	optionsHtml += "<option class='bg-white' value='bge-m3'>bge-m3</option>"
	for _, model := range v.models {
		optionsHtml += fmt.Sprintf("<option class='bg-white' value='%s'>%s</option>", model, model)
	}
	v.optionsHtml = optionsHtml
	return v
}

func (v *View) RenderIndex(w http.ResponseWriter) {
	v.renderOptions()
	codeHtml := fmt.Sprintf(indexHTML, v.optionsHtml)
	w.Header().Set("Content-Type", "text/html")
	w.Write([]byte(codeHtml))
}
