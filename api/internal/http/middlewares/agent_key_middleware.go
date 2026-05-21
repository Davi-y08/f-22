package middlewares

import (
	"context"
	"errors"
	"net/http"
	"strings"

	domainAgent "stealth-lens/internal/domain/agent"
	"stealth-lens/internal/httpx"
)

type AgentAccessKeyAuthenticator interface {
	AuthenticateAgentAccessKey(ctx context.Context, plainKey string) (*domainAgent.AgentAccessKey, error)
}

const agentAccessKeyIDContextKey contextKey = "agent_access_key_id"

func AgentKeyAuthMiddleware(authenticator AgentAccessKeyAuthenticator, next AppHandler) AppHandler {
	return func(w http.ResponseWriter, r *http.Request) error {
		plainKey := strings.TrimSpace(r.Header.Get("X-Agent-Key"))
		if plainKey == "" {
			plainKey = strings.TrimSpace(r.Header.Get("X-API-Key"))
		}
		if plainKey == "" {
			return httpx.Unauthorized(errors.New("missing agent access key"))
		}

		accessKey, err := authenticator.AuthenticateAgentAccessKey(r.Context(), plainKey)
		if err != nil {
			return httpx.Unauthorized(errors.New("invalid agent access key"))
		}

		ctx := context.WithValue(r.Context(), userIDContextKey, accessKey.UserID)
		ctx = context.WithValue(ctx, agentAccessKeyIDContextKey, accessKey.ID)
		return next(w, r.WithContext(ctx))
	}
}
