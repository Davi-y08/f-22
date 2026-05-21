package handlers

import (
	"encoding/json"
	"errors"
	"net/http"

	appCamera "stealth-lens/internal/application/camera"
	mapErrors "stealth-lens/internal/http/http_errors"
	"stealth-lens/internal/http/middlewares"
	"stealth-lens/internal/httpx"
)

type AgentCameraHandler struct {
	service *appCamera.CameraService
}

func NewAgentCameraHandler(service *appCamera.CameraService) *AgentCameraHandler {
	return &AgentCameraHandler{service: service}
}

func (h *AgentCameraHandler) SyncCamerasHandler(w http.ResponseWriter, r *http.Request) error {
	userID, err := middlewares.UserIDFromContext(r.Context())
	if err != nil {
		return httpx.Unauthorized(err)
	}

	var dto appCamera.SyncAgentCamerasDto
	if err := json.NewDecoder(r.Body).Decode(&dto); err != nil {
		return httpx.BadRequest(errors.New("corpo invalido"))
	}

	result, err := h.service.SyncAgentCamerasService(r.Context(), userID, dto)
	if err != nil {
		return mapErrors.MapErrorsCamera(err)
	}

	return json.NewEncoder(w).Encode(result)
}
