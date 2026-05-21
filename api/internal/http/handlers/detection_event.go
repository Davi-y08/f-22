package handlers

import (
	"encoding/json"
	"net/http"
	"strconv"

	appDetection "stealth-lens/internal/application/detection"
	mapErrors "stealth-lens/internal/http/http_errors"
	"stealth-lens/internal/http/middlewares"
	"stealth-lens/internal/httpx"
)

type DetectionEventHandler struct {
	service *appDetection.DetectionEventService
}

func NewDetectionEventHandler(service *appDetection.DetectionEventService) *DetectionEventHandler {
	return &DetectionEventHandler{service: service}
}

func (h *DetectionEventHandler) CreateAgentDetectionEventHandler(w http.ResponseWriter, r *http.Request) error {
	userID, err := middlewares.UserIDFromContext(r.Context())
	if err != nil {
		return httpx.Unauthorized(err)
	}

	var dto appDetection.AgentDetectionEventDto
	if err := json.NewDecoder(r.Body).Decode(&dto); err != nil {
		return httpx.BadRequest(err)
	}

	event, err := h.service.CreateFromAgent(r.Context(), userID, dto)
	if err != nil {
		return mapErrors.MapErrorsDetectionEvent(err)
	}

	w.WriteHeader(http.StatusCreated)
	return json.NewEncoder(w).Encode(event)
}

func (h *DetectionEventHandler) ListDetectionEventsHandler(w http.ResponseWriter, r *http.Request) error {
	userID, err := middlewares.UserIDFromContext(r.Context())
	if err != nil {
		return httpx.Unauthorized(err)
	}

	limit := 100
	if rawLimit := r.URL.Query().Get("limit"); rawLimit != "" {
		parsedLimit, parseErr := strconv.Atoi(rawLimit)
		if parseErr == nil {
			limit = parsedLimit
		}
	}

	events, err := h.service.ListByUser(r.Context(), userID, limit)
	if err != nil {
		return mapErrors.MapErrorsDetectionEvent(err)
	}

	return json.NewEncoder(w).Encode(events)
}
