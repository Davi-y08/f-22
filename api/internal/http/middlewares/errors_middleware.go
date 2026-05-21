package middlewares

import (
	"encoding/json"
	"errors"
	"net/http"

	"stealth-lens/internal/httpx"
)

func ErrorsMiddleware(next AppHandler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		if err := next(w, r); err != nil {
			var appErr *httpx.AppError
			if errors.As(err, &appErr) {
				w.WriteHeader(appErr.Status)
				_ = json.NewEncoder(w).Encode(map[string]string{
					"error": appErr.Message,
				})
				return
			}

			w.WriteHeader(http.StatusInternalServerError)
			_ = json.NewEncoder(w).Encode(map[string]string{
				"error": "internal error",
			})
		}
	}
}
