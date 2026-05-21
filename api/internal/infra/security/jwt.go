package security

import (
	"errors"
	"log"
	"os"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
	"github.com/joho/godotenv"
)

var jwtKey []byte

func LoadJWTConfig() {
	_ = godotenv.Load(".env")

	secret := os.Getenv("JWT_SECRET_KEY")
	if secret == "" {
		log.Fatal("JWT_SECRET_KEY nao configurada")
	}

	jwtKey = []byte(secret)
}

func SigningKey() []byte {
	return jwtKey
}

func GenerateTokenJWT(userID uuid.UUID) (string, error) {
	if len(jwtKey) == 0 {
		return "", errors.New("jwt key nao configurada")
	}

	claims := jwt.RegisteredClaims{
		Subject:   userID.String(),
		Issuer:    "api",
		IssuedAt:  jwt.NewNumericDate(time.Now()),
		ExpiresAt: jwt.NewNumericDate(time.Now().Add(24 * time.Hour)),
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	tokenString, err := token.SignedString(jwtKey)
	if err != nil {
		return "", errors.New("erro ao gerar token jwt")
	}

	return tokenString, nil
}
