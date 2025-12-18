package main

import (
	"net/http"
)

func ConcurrencyLimiter(maxConcurrent int) func(next http.Handler) http.Handler {
	semaphore := make(chan struct{}, maxConcurrent)

	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			select {
			case semaphore <- struct{}{}:
				defer func() { <-semaphore }()
				next.ServeHTTP(w, r)
			case <-r.Context().Done():
				return
			default:
				respondError(w, http.StatusTooManyRequests, "rate_limited", "Server is busy, please try again later")
			}
		})
	}
}