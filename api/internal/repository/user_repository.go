package repository

import (
	"context"
	"errors"

	"github.com/google/uuid"
	"gorm.io/gorm"

	"stealth-lens/internal/domain/user"
)

type UserRepository struct {
	db *gorm.DB
}

func NewUserRepository(db *gorm.DB) *UserRepository {
	return &UserRepository{db: db}
}

func (r *UserRepository) CreateUser(ctx context.Context, foundUser *user.User) error {
	return r.db.WithContext(ctx).Model(&user.User{}).Create(foundUser).Error
}

func (r *UserRepository) GetUserByID(ctx context.Context, id uuid.UUID) (*user.User, error) {
	var foundUser user.User
	if err := r.db.WithContext(ctx).Model(&user.User{}).First(&foundUser, id).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, gorm.ErrRecordNotFound
		}

		return nil, err
	}

	return &foundUser, nil
}

func (r *UserRepository) GetUserByEmail(ctx context.Context, email string) (*user.User, error) {
	var foundUser user.User
	if err := r.db.WithContext(ctx).Model(&user.User{}).Where("email = ?", email).First(&foundUser).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, gorm.ErrRecordNotFound
		}

		return nil, err
	}

	return &foundUser, nil
}
