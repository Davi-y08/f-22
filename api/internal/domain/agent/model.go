package agent

import (
	"time"

	"github.com/google/uuid"

	"stealth-lens/internal/domain/user"
)

type AgentAccessKey struct {
	ID         uuid.UUID  `json:"id" gorm:"type:uuid;default:gen_random_uuid();primaryKey"`
	Name       string     `json:"name" gorm:"size:120;not null"`
	KeyHash    string     `json:"-" gorm:"size:64;uniqueIndex;not null"`
	KeyPrefix  string     `json:"key_prefix" gorm:"size:24;index;not null"`
	UserID     uuid.UUID  `json:"-" gorm:"type:uuid;not null;index"`
	User       user.User  `json:"-" gorm:"foreignKey:UserID;references:ID;constraint:OnUpdate:CASCADE,OnDelete:CASCADE"`
	LastUsedAt *time.Time `json:"last_used_at,omitempty"`
	RevokedAt  *time.Time `json:"revoked_at,omitempty"`
	CreatedAt  time.Time  `json:"created_at"`
	UpdatedAt  time.Time  `json:"updated_at"`
}
