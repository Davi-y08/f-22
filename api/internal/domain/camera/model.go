package camera

import (
	"time"

	"github.com/google/uuid"

	"stealth-lens/internal/domain/user"
)

type Camera struct {
	ID         uuid.UUID  `json:"id" gorm:"type:uuid;default:gen_random_uuid();primaryKey"`
	Name       string     `json:"name" gorm:"size:120;not null"`
	Location   string     `json:"location" gorm:"size:180;not null"`
	URL        string     `json:"url" gorm:"size:512;not null"`
	Status     string     `json:"status" gorm:"size:32;not null;default:unknown"`
	AgentID    string     `json:"agent_id,omitempty" gorm:"size:120;index"`
	ExternalID string     `json:"external_id,omitempty" gorm:"size:180;index"`
	LastSeenAt *time.Time `json:"last_seen_at,omitempty"`
	UserID     uuid.UUID  `json:"-" gorm:"type:uuid;not null;index"`
	User       user.User  `json:"-" gorm:"foreignKey:UserID;references:ID;constraint:OnUpdate:CASCADE,OnDelete:CASCADE"`
	CreatedAt  time.Time  `json:"created_at"`
	UpdatedAt  time.Time  `json:"updated_at"`
}
