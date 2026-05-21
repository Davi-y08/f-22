package database

import (
	"log"

	"stealth-lens/internal/domain/agent"
	"stealth-lens/internal/domain/camera"
	"stealth-lens/internal/domain/detection"
	"stealth-lens/internal/domain/user"

	"gorm.io/gorm"
)

func RunMigrations(db *gorm.DB) {
	if err := db.Exec("CREATE EXTENSION IF NOT EXISTS pgcrypto").Error; err != nil {
		log.Fatal("erro ao habilitar extensao pgcrypto")
	}

	if err := db.AutoMigrate(&user.User{}, &agent.AgentAccessKey{}, &camera.Camera{}, &detection.DetectionEvent{}); err != nil {
		log.Fatal("erro ao executar migrations")
	}
}
