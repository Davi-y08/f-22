package handlers

import (
	"encoding/json"
	"errors"
	"net/http"

	"github.com/google/uuid"

	appAgentKey "stealth-lens/internal/application/agentkey"
	mapErrors "stealth-lens/internal/http/http_errors"
	"stealth-lens/internal/http/middlewares"
	"stealth-lens/internal/httpx"
)

type AgentAccessKeyHandler struct {
	service *appAgentKey.AgentAccessKeyService
}

func NewAgentAccessKeyHandler(service *appAgentKey.AgentAccessKeyService) *AgentAccessKeyHandler {
	return &AgentAccessKeyHandler{service: service}
}

func (h *AgentAccessKeyHandler) CreateAgentAccessKeyHandler(w http.ResponseWriter, r *http.Request) error {
	userID, err := middlewares.UserIDFromContext(r.Context())
	if err != nil {
		return httpx.Unauthorized(err)
	}

	var dto appAgentKey.CreateAgentAccessKeyDto
	if err := json.NewDecoder(r.Body).Decode(&dto); err != nil {
		return httpx.BadRequest(errors.New("corpo invalido"))
	}

	createdKey, err := h.service.CreateAgentAccessKey(r.Context(), userID, dto)
	if err != nil {
		return mapErrors.MapErrorsAgentAccessKey(err)
	}

	w.WriteHeader(http.StatusCreated)
	return json.NewEncoder(w).Encode(createdKey)
}

func (h *AgentAccessKeyHandler) ListAgentAccessKeysHandler(w http.ResponseWriter, r *http.Request) error {
	userID, err := middlewares.UserIDFromContext(r.Context())
	if err != nil {
		return httpx.Unauthorized(err)
	}

	keys, err := h.service.ListAgentAccessKeys(r.Context(), userID)
	if err != nil {
		return mapErrors.MapErrorsAgentAccessKey(err)
	}

	return json.NewEncoder(w).Encode(keys)
}

func (h *AgentAccessKeyHandler) RevokeAgentAccessKeyHandler(w http.ResponseWriter, r *http.Request) error {
	userID, err := middlewares.UserIDFromContext(r.Context())
	if err != nil {
		return httpx.Unauthorized(err)
	}

	keyID, err := uuid.Parse(r.PathValue("id"))
	if err != nil {
		return httpx.BadRequest(errors.New("id da chave invalido"))
	}

	if err := h.service.RevokeAgentAccessKey(r.Context(), keyID, userID); err != nil {
		return mapErrors.MapErrorsAgentAccessKey(err)
	}

	w.WriteHeader(http.StatusNoContent)
	return nil
}
