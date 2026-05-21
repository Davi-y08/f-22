package agentkey

import (
	"strings"
	"time"

	"github.com/google/uuid"

	domainAgent "stealth-lens/internal/domain/agent"
)

type CreateAgentAccessKeyDto struct {
	Name string `json:"name"`
}

type AgentAccessKeyView struct {
	ID         uuid.UUID  `json:"id"`
	Name       string     `json:"name"`
	KeyPrefix  string     `json:"key_prefix"`
	LastUsedAt *time.Time `json:"last_used_at,omitempty"`
	RevokedAt  *time.Time `json:"revoked_at,omitempty"`
	CreatedAt  time.Time  `json:"created_at"`
}

type CreatedAgentAccessKeyView struct {
	AgentAccessKeyView
	AccessKey string `json:"access_key"`
}

func validateCreateDto(dto CreateAgentAccessKeyDto) (string, error) {
	name := strings.TrimSpace(dto.Name)
	if name == "" {
		name = "Distribuido Stealth Lens"
	}

	if len(name) > 120 {
		return "", domainAgent.ErrAgentAccessKeyInvalidData
	}

	return name, nil
}

func toView(accessKey domainAgent.AgentAccessKey) AgentAccessKeyView {
	return AgentAccessKeyView{
		ID:         accessKey.ID,
		Name:       accessKey.Name,
		KeyPrefix:  accessKey.KeyPrefix,
		LastUsedAt: accessKey.LastUsedAt,
		RevokedAt:  accessKey.RevokedAt,
		CreatedAt:  accessKey.CreatedAt,
	}
}
