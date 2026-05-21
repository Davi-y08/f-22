package http

import (
	"net/http"

	agentKeyApp "stealth-lens/internal/application/agentkey"
	cameraApp "stealth-lens/internal/application/camera"
	detectionApp "stealth-lens/internal/application/detection"
	userApp "stealth-lens/internal/application/user"
	"stealth-lens/internal/http/handlers"
	"stealth-lens/internal/http/middlewares"
	repo "stealth-lens/internal/repository"

	"gorm.io/gorm"
)

func SetupRouter(db *gorm.DB) http.Handler {
	mux := http.NewServeMux()

	userRepo := repo.NewUserRepository(db)
	userService := userApp.NewUserService(userRepo)
	userHandler := handlers.NewUserHandler(userService)

	cameraRepo := repo.NewCameraRepository(db)
	cameraService := cameraApp.NewCameraService(cameraRepo)
	cameraHandler := handlers.NewCameraHandler(cameraService)
	agentCameraHandler := handlers.NewAgentCameraHandler(cameraService)

	agentAccessKeyRepo := repo.NewAgentAccessKeyRepository(db)
	agentAccessKeyService := agentKeyApp.NewAgentAccessKeyService(agentAccessKeyRepo)
	agentAccessKeyHandler := handlers.NewAgentAccessKeyHandler(agentAccessKeyService)

	detectionEventRepo := repo.NewDetectionEventRepository(db)
	detectionEventService := detectionApp.NewDetectionEventService(detectionEventRepo, cameraRepo)
	detectionEventHandler := handlers.NewDetectionEventHandler(detectionEventService)

	mux.HandleFunc("GET /ping", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("pong"))
	})

	mux.HandleFunc("POST /users", middlewares.ErrorsMiddleware(userHandler.CreateUserHandler))
	mux.HandleFunc("POST /users/login", middlewares.ErrorsMiddleware(userHandler.LoginHandler))
	mux.HandleFunc("GET /users/me", middlewares.ErrorsMiddleware(middlewares.JWTAuthMiddleware(userHandler.MeHandler)))

	mux.HandleFunc("POST /agent-keys", middlewares.ErrorsMiddleware(middlewares.JWTAuthMiddleware(agentAccessKeyHandler.CreateAgentAccessKeyHandler)))
	mux.HandleFunc("GET /agent-keys", middlewares.ErrorsMiddleware(middlewares.JWTAuthMiddleware(agentAccessKeyHandler.ListAgentAccessKeysHandler)))
	mux.HandleFunc("DELETE /agent-keys/{id}", middlewares.ErrorsMiddleware(middlewares.JWTAuthMiddleware(agentAccessKeyHandler.RevokeAgentAccessKeyHandler)))

	mux.HandleFunc("POST /cameras", middlewares.ErrorsMiddleware(middlewares.JWTAuthMiddleware(cameraHandler.CreateCameraHandler)))
	mux.HandleFunc("GET /cameras", middlewares.ErrorsMiddleware(middlewares.JWTAuthMiddleware(cameraHandler.ListCamerasHandler)))
	mux.HandleFunc("GET /cameras/{id}", middlewares.ErrorsMiddleware(middlewares.JWTAuthMiddleware(cameraHandler.GetCameraHandler)))
	mux.HandleFunc("PUT /cameras/{id}", middlewares.ErrorsMiddleware(middlewares.JWTAuthMiddleware(cameraHandler.UpdateCameraHandler)))
	mux.HandleFunc("DELETE /cameras/{id}", middlewares.ErrorsMiddleware(middlewares.JWTAuthMiddleware(cameraHandler.DeleteCameraHandler)))

	mux.HandleFunc("GET /events", middlewares.ErrorsMiddleware(middlewares.JWTAuthMiddleware(detectionEventHandler.ListDetectionEventsHandler)))
	mux.HandleFunc("POST /agent/cameras/sync", middlewares.ErrorsMiddleware(middlewares.AgentKeyAuthMiddleware(agentAccessKeyService, agentCameraHandler.SyncCamerasHandler)))
	mux.HandleFunc("POST /agent/events", middlewares.ErrorsMiddleware(middlewares.AgentKeyAuthMiddleware(agentAccessKeyService, detectionEventHandler.CreateAgentDetectionEventHandler)))

	return middlewares.CORSMiddleware(mux)
}
