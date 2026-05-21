package user

import (
	"net/mail"
	"strings"

	domainUser "stealth-lens/internal/domain/user"
	"stealth-lens/internal/infra/security"
)

type CreateUserDto struct {
	Name            string `json:"name"`
	Email           string `json:"email"`
	Password        string `json:"password"`
	ConfirmPassword string `json:"confirm_password"`
}

type LoginDto struct {
	Email    string `json:"email"`
	Password string `json:"password"`
}

func validateEmail(value string) bool {
	_, err := mail.ParseAddress(value)
	return err == nil
}

func ValidateDto(dto CreateUserDto) (*domainUser.User, error) {
	name := strings.TrimSpace(dto.Name)
	email := strings.TrimSpace(strings.ToLower(dto.Email))

	if !validateEmail(email) || len(name) < 5 || len(dto.Password) < 6 {
		return nil, domainUser.ErrUserInvalidData
	}

	if dto.ConfirmPassword != dto.Password {
		return nil, domainUser.ErrPasswordDontMatch
	}

	hash, err := security.HashPassword(dto.Password)
	if err != nil {
		return nil, domainUser.ErrHashedPassword
	}

	return &domainUser.User{
		Email:        email,
		PasswordHash: hash,
		Name:         name,
		Role:         domainUser.DefaultRole,
	}, nil
}
