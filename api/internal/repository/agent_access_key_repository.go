package repository

import (
	"context"
	"errors"
	"time"

	"github.com/google/uuid"
	"gorm.io/gorm"

	"stealth-lens/internal/domain/agent"
)

type AgentAccessKeyRepository struct {
	db *gorm.DB
}

func NewAgentAccessKeyRepository(db *gorm.DB) *AgentAccessKeyRepository {
	return &AgentAccessKeyRepository{db: db}
}

func (r *AgentAccessKeyRepository) CreateAgentAccessKey(ctx context.Context, accessKey *agent.AgentAccessKey) error {
	return r.db.WithContext(ctx).Model(&agent.AgentAccessKey{}).Create(accessKey).Error
}

func (r *AgentAccessKeyRepository) GetAgentAccessKeysByUser(ctx context.Context, userID uuid.UUID) ([]agent.AgentAccessKey, error) {
	var keys []agent.AgentAccessKey
	if err := r.db.WithContext(ctx).Model(&agent.AgentAccessKey{}).Where("user_id = ?", userID).Order("created_at DESC").Find(&keys).Error; err != nil {
		return nil, err
	}

	return keys, nil
}

func (r *AgentAccessKeyRepository) GetActiveAgentAccessKeyByHash(ctx context.Context, keyHash string) (*agent.AgentAccessKey, error) {
	var accessKey agent.AgentAccessKey
	if err := r.db.WithContext(ctx).Model(&agent.AgentAccessKey{}).Where("key_hash = ? AND revoked_at IS NULL", keyHash).First(&accessKey).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, gorm.ErrRecordNotFound
		}

		return nil, err
	}

	return &accessKey, nil
}

func (r *AgentAccessKeyRepository) TouchAgentAccessKey(ctx context.Context, keyID uuid.UUID, lastUsedAt time.Time) error {
	return r.db.WithContext(ctx).Model(&agent.AgentAccessKey{}).Where("id = ?", keyID).Update("last_used_at", lastUsedAt).Error
}

func (r *AgentAccessKeyRepository) RevokeAgentAccessKey(ctx context.Context, keyID, userID uuid.UUID) error {
	now := time.Now().UTC()
	result := r.db.WithContext(ctx).Model(&agent.AgentAccessKey{}).
		Where("id = ? AND user_id = ? AND revoked_at IS NULL", keyID, userID).
		Update("revoked_at", now)
	if result.Error != nil {
		return result.Error
	}
	if result.RowsAffected == 0 {
		return gorm.ErrRecordNotFound
	}

	return nil
}
