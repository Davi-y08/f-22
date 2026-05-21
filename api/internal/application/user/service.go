package user

import (
	"context"
	"errors"
	"strings"

	"github.com/google/uuid"

	shared "stealth-lens/internal/application"
	domainUser "stealth-lens/internal/domain/user"
	"stealth-lens/internal/infra/security"
	repo "stealth-lens/internal/repository"

	"gorm.io/gorm"
)

type UserService struct {
	repo *repo.UserRepository
}

func NewUserService(repo *repo.UserRepository) *UserService {
	return &UserService{repo: repo}
}

func (s *UserService) CreateUser(ctx context.Context, dto CreateUserDto) error {
	newUser, err := ValidateDto(dto)
	if err != nil {
		return err
	}

	existing, err := s.repo.GetUserByEmail(ctx, newUser.Email)
	switch {
	case err == nil && existing != nil:
		return domainUser.ErrExistingUser
	case err != nil && !errors.Is(err, gorm.ErrRecordNotFound):
		return shared.ErrInDataBase
	}

	if err := s.repo.CreateUser(ctx, newUser); err != nil {
		return shared.ErrInDataBase
	}

	return nil
}

func (s *UserService) GetUserByID(ctx context.Context, userID uuid.UUID) (*domainUser.User, error) {
	foundUser, err := s.repo.GetUserByID(ctx, userID)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, domainUser.ErrUserNotFound
		}

		return nil, shared.ErrInDataBase
	}

	return foundUser, nil
}

func (s *UserService) Login(ctx context.Context, dto LoginDto) (*domainUser.User, error) {
	email := strings.TrimSpace(strings.ToLower(dto.Email))
	if email == "" || dto.Password == "" {
		return nil, domainUser.ErrInvalidCredentials
	}

	foundUser, err := s.repo.GetUserByEmail(ctx, email)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, domainUser.ErrInvalidCredentials
		}

		return nil, shared.ErrInDataBase
	}

	if !security.CheckPassword(foundUser.PasswordHash, dto.Password) {
		return nil, domainUser.ErrInvalidCredentials
	}

	return foundUser, nil
}
