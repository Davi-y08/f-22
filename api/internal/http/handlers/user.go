package handlers

import (
	"encoding/json"
	"errors"
	"net/http"

	appUser "stealth-lens/internal/application/user"
	mapErrors "stealth-lens/internal/http/http_errors"
	"stealth-lens/internal/http/middlewares"
	"stealth-lens/internal/httpx"
	"stealth-lens/internal/infra/security"
)

type UserHandler struct {
	service *appUser.UserService
}

func NewUserHandler(service *appUser.UserService) *UserHandler {
	return &UserHandler{service: service}
}

func (h *UserHandler) CreateUserHandler(w http.ResponseWriter, r *http.Request) error {
	if r.Method != http.MethodPost {
		return httpx.MethodNotAllowed(errors.New("method not allowed"))
	}

	var dto appUser.CreateUserDto
	if err := json.NewDecoder(r.Body).Decode(&dto); err != nil {
		return httpx.BadRequest(errors.New("corpo invalido"))
	}

	if err := h.service.CreateUser(r.Context(), dto); err != nil {
		return mapErrors.MapErrorsUser(err)
	}

	w.WriteHeader(http.StatusCreated)
	return json.NewEncoder(w).Encode(map[string]string{
		"message": "user created",
	})
}

func (h *UserHandler) MeHandler(w http.ResponseWriter, r *http.Request) error {
	userID, err := middlewares.UserIDFromContext(r.Context())
	if err != nil {
		return httpx.Unauthorized(err)
	}

	loggedUser, err := h.service.GetUserByID(r.Context(), userID)
	if err != nil {
		return mapErrors.MapErrorsUser(err)
	}

	return json.NewEncoder(w).Encode(loggedUser)
}
func (h *UserHandler) LoginHandler(w http.ResponseWriter, r *http.Request) error {
	if r.Method != http.MethodPost {
		return httpx.MethodNotAllowed(errors.New("method not allowed"))
	}

	var dto appUser.LoginDto
	if err := json.NewDecoder(r.Body).Decode(&dto); err != nil {
		return httpx.BadRequest(errors.New("corpo invalido"))
	}

	loggedUser, err := h.service.Login(r.Context(), dto)
	if err != nil {
		return mapErrors.MapErrorsUser(err)
	}

	token, err := security.GenerateTokenJWT(loggedUser.ID)
	if err != nil {
		return httpx.Internal(err)
	}

	http.SetCookie(w, &http.Cookie{
		Name:     "access_token",
		Value:    token,
		Path:     "/",
		MaxAge:   60 * 60 * 12,
		HttpOnly: true,
		SameSite: http.SameSiteLaxMode,
	})

	w.WriteHeader(http.StatusOK)
	return json.NewEncoder(w).Encode(map[string]string{
		"message": "login realizado com sucesso",
	})
}
