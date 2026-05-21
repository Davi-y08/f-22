package database

import (
	"log"
	"os"

	"github.com/joho/godotenv"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
)

func LoadDbConfigs() string {
	_ = godotenv.Load(".env")

	credentials := os.Getenv("DATABASE_URL")
	if credentials == "" {
		log.Fatal("DATABASE_URL nao configurada")
	}

	return credentials
}

func Connect() *gorm.DB {
	dsn := LoadDbConfigs()
	db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{})
	if err != nil {
		log.Fatal("ocorreu um erro ao se conectar com o banco de dados")
	}

	return db
}
