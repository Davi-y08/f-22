package agentkey

import (
	"context"
	"errors"
	"strings"
	"time"

	"github.com/google/uuid"
	"gorm.io/gorm"

	shared "stealth-lens/internal/application"
	domainAgent "stealth-lens/internal/domain/agent"
	"stealth-lens/internal/infra/security"
	repo "stealth-lens/internal/repository"
)

type AgentAccessKeyService struct {
	repo *repo.AgentAccessKeyRepository
}

func NewAgentAccessKeyService(repo *repo.AgentAccessKeyRepository) *AgentAccessKeyService {
	return &AgentAccessKeyService{repo: repo}
}

func (s *AgentAccessKeyService) CreateAgentAccessKey(ctx context.Context, userID uuid.UUID, dto CreateAgentAccessKeyDto) (*CreatedAgentAccessKeyView, error) {
	name, err := validateCreateDto(dto)
	if err != nil {
		return nil, err
	}

	plainKey, err := security.GenerateAgentAccessKey()
	if err != nil {
		return nil, shared.ErrInDataBase
	}

	accessKey := &domainAgent.AgentAccessKey{
		Name:      name,
		KeyHash:   security.HashAgentAccessKey(plainKey),
		KeyPrefix: security.AgentAccessKeyPrefix(plainKey),
		UserID:    userID,
	}

	if err := s.repo.CreateAgentAccessKey(ctx, accessKey); err != nil {
		return nil, shared.ErrInDataBase
	}

	return &CreatedAgentAccessKeyView{
		AgentAccessKeyView: toView(*accessKey),
		AccessKey:          plainKey,
	}, nil
}

func (s *AgentAccessKeyService) ListAgentAccessKeys(ctx context.Context, userID uuid.UUID) ([]AgentAccessKeyView, error) {
	keys, err := s.repo.GetAgentAccessKeysByUser(ctx, userID)
	if err != nil {
		return nil, shared.ErrInDataBase
	}

	views := make([]AgentAccessKeyView, 0, len(keys))
	for _, key := range keys {
		views = append(views, toView(key))
	}

	return views, nil
}

func (s *AgentAccessKeyService) RevokeAgentAccessKey(ctx context.Context, keyID, userID uuid.UUID) error {
	if err := s.repo.RevokeAgentAccessKey(ctx, keyID, userID); err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return domainAgent.ErrAgentAccessKeyNotFound
		}

		return shared.ErrInDataBase
	}

	return nil
}

func (s *AgentAccessKeyService) AuthenticateAgentAccessKey(ctx context.Context, plainKey string) (*domainAgent.AgentAccessKey, error) {
	plainKey = strings.TrimSpace(plainKey)
	if !security.IsValidAgentAccessKeyFormat(plainKey) {
		return nil, domainAgent.ErrAgentAccessKeyInvalid
	}

	accessKey, err := s.repo.GetActiveAgentAccessKeyByHash(ctx, security.HashAgentAccessKey(plainKey))
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, domainAgent.ErrAgentAccessKeyInvalid
		}

		return nil, shared.ErrInDataBase
	}

	now := time.Now().UTC()
	accessKey.LastUsedAt = &now
	if err := s.repo.TouchAgentAccessKey(ctx, accessKey.ID, now); err != nil {
		return nil, shared.ErrInDataBase
	}

	return accessKey, nil
}
