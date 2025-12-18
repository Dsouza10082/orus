package main

import (
	"log/slog"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5/middleware"
)

func RequestLogger(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		ww := middleware.NewWrapResponseWriter(w, r.ProtoMajor)

		defer func() {
			go func(method, path string, status int, duration time.Duration) {
				slog.Info("request",
					"method", method,
					"path", path,
					"status", status,
					"duration", duration,
				)
			}(r.Method, r.URL.Path, ww.Status(), time.Since(start))
		}()

		next.ServeHTTP(ww, r)
	})
}