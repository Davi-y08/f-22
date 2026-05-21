package main

import (
	"log"
	"net/http"

	httpServer "stealth-lens/internal/http"
	db "stealth-lens/internal/infra/database"
	"stealth-lens/internal/infra/security"
)

func main() {
	database := db.Connect()
	db.RunMigrations(database)
	security.LoadJWTConfig()

	router := httpServer.SetupRouter(database)

	log.Println("servidor rodando na porta :8080")
	if err := http.ListenAndServe(":8080", router); err != nil {
		log.Fatal("erro ao inicializar servidor: ", err.Error())
	}
}
