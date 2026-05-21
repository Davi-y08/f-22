package middlewares

import (
	"context"
	"errors"
	"net/http"

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"

	"stealth-lens/internal/httpx"
	"stealth-lens/internal/infra/security"
)

type contextKey string

const userIDContextKey contextKey = "user_id"

func JWTAuthMiddleware(next AppHandler) AppHandler {
	return func(w http.ResponseWriter, r *http.Request) error {
		cookie, err := r.Cookie("access_token")
		if err != nil {
			return httpx.Unauthorized(errors.New("missing authentication token"))
		}

		claims := &jwt.RegisteredClaims{}
		token, err := jwt.ParseWithClaims(
			cookie.Value,
			claims,
			func(token *jwt.Token) (interface{}, error) {
				if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
					return nil, jwt.ErrSignatureInvalid
				}

				return security.SigningKey(), nil
			},
			jwt.WithIssuer("api"),
		)
		if err != nil || !token.Valid {
			return httpx.Unauthorized(errors.New("invalid or expired token"))
		}

		userID, err := uuid.Parse(claims.Subject)
		if err != nil {
			return httpx.Unauthorized(errors.New("invalid token subject"))
		}

		ctx := context.WithValue(r.Context(), userIDContextKey, userID)
		return next(w, r.WithContext(ctx))
	}
}

func UserIDFromContext(ctx context.Context) (uuid.UUID, error) {
	value := ctx.Value(userIDContextKey)
	userID, ok := value.(uuid.UUID)
	if !ok || userID == uuid.Nil {
		return uuid.Nil, errors.New("user not found in context")
	}

	return userID, nil
}
