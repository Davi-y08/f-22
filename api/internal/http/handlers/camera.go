package handlers

import (
	"encoding/json"
	"errors"
	"net/http"

	"github.com/google/uuid"

	appCamera "stealth-lens/internal/application/camera"
	mapErrors "stealth-lens/internal/http/http_errors"
	"stealth-lens/internal/http/middlewares"
	"stealth-lens/internal/httpx"
)

type CameraHandler struct {
	service *appCamera.CameraService
}

func NewCameraHandler(service *appCamera.CameraService) *CameraHandler {
	return &CameraHandler{service: service}
}

func (h *CameraHandler) CreateCameraHandler(w http.ResponseWriter, r *http.Request) error {
	var dto appCamera.CreateCameraDto
	if err := json.NewDecoder(r.Body).Decode(&dto); err != nil {
		return httpx.BadRequest(errors.New("corpo invalido"))
	}

	userID, err := middlewares.UserIDFromContext(r.Context())
	if err != nil {
		return httpx.Unauthorized(err)
	}
	dto.UserID = userID

	createdCamera, err := h.service.CreateCameraService(r.Context(), dto)
	if err != nil {
		return mapErrors.MapErrorsCamera(err)
	}

	w.WriteHeader(http.StatusCreated)
	return json.NewEncoder(w).Encode(createdCamera)
}

func (h *CameraHandler) ListCamerasHandler(w http.ResponseWriter, r *http.Request) error {
	userID, err := middlewares.UserIDFromContext(r.Context())
	if err != nil {
		return httpx.Unauthorized(err)
	}

	cameras, err := h.service.ListCamerasService(r.Context(), userID)
	if err != nil {
		return mapErrors.MapErrorsCamera(err)
	}

	return json.NewEncoder(w).Encode(cameras)
}

func (h *CameraHandler) GetCameraHandler(w http.ResponseWriter, r *http.Request) error {
	cameraID, userID, err := parseCameraRequestIDs(r)
	if err != nil {
		return err
	}

	foundCamera, err := h.service.GetCameraByIDService(r.Context(), cameraID, userID)
	if err != nil {
		return mapErrors.MapErrorsCamera(err)
	}

	return json.NewEncoder(w).Encode(foundCamera)
}

func (h *CameraHandler) UpdateCameraHandler(w http.ResponseWriter, r *http.Request) error {
	cameraID, userID, err := parseCameraRequestIDs(r)
	if err != nil {
		return err
	}

	var dto appCamera.UpdateCameraDto
	if err := json.NewDecoder(r.Body).Decode(&dto); err != nil {
		return httpx.BadRequest(errors.New("corpo invalido"))
	}

	updatedCamera, err := h.service.UpdateCameraService(r.Context(), cameraID, userID, dto)
	if err != nil {
		return mapErrors.MapErrorsCamera(err)
	}

	return json.NewEncoder(w).Encode(updatedCamera)
}

func (h *CameraHandler) DeleteCameraHandler(w http.ResponseWriter, r *http.Request) error {
	cameraID, userID, err := parseCameraRequestIDs(r)
	if err != nil {
		return err
	}

	if err := h.service.DeleteCameraService(r.Context(), cameraID, userID); err != nil {
		return mapErrors.MapErrorsCamera(err)
	}

	w.WriteHeader(http.StatusNoContent)
	return nil
}

func parseCameraRequestIDs(r *http.Request) (uuid.UUID, uuid.UUID, error) {
	cameraID, err := uuid.Parse(r.PathValue("id"))
	if err != nil {
		return uuid.Nil, uuid.Nil, httpx.BadRequest(errors.New("camera id invalido"))
	}

	userID, err := middlewares.UserIDFromContext(r.Context())
	if err != nil {
		return uuid.Nil, uuid.Nil, httpx.Unauthorized(err)
	}

	return cameraID, userID, nil
}
