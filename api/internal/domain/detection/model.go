package detection

import (
	"time"

	"github.com/google/uuid"

	"stealth-lens/internal/domain/camera"
	"stealth-lens/internal/domain/user"
)

type DetectionEvent struct {
	ID               uuid.UUID      `json:"id" gorm:"type:uuid;default:gen_random_uuid();primaryKey"`
	UserID           uuid.UUID      `json:"-" gorm:"type:uuid;not null;index"`
	User             user.User      `json:"-" gorm:"foreignKey:UserID;references:ID;constraint:OnUpdate:CASCADE,OnDelete:CASCADE"`
	CameraID         *uuid.UUID     `json:"camera_id,omitempty" gorm:"type:uuid;index"`
	Camera           *camera.Camera `json:"-" gorm:"foreignKey:CameraID;references:ID;constraint:OnUpdate:CASCADE,OnDelete:SET NULL"`
	AgentID          string         `json:"agent_id" gorm:"size:120;index;not null"`
	LocalEventID     string         `json:"local_event_id" gorm:"size:120;index"`
	CameraExternalID string         `json:"camera_external_id" gorm:"size:180;index"`
	CameraName       string         `json:"camera_name" gorm:"size:180"`
	EventType        string         `json:"event_type" gorm:"size:80;index;not null"`
	Confidence       float64        `json:"confidence"`
	ModelAlias       string         `json:"model_alias" gorm:"size:120"`
	Label            string         `json:"label" gorm:"size:120"`
	BBox             string         `json:"bbox" gorm:"type:text"`
	FrameSize        string         `json:"frame_size,omitempty" gorm:"type:text"`
	Zone             *string        `json:"zone,omitempty" gorm:"size:120"`
	SnapshotPath     *string        `json:"snapshot_path,omitempty" gorm:"size:512"`
	Metadata         string         `json:"metadata,omitempty" gorm:"type:text"`
	OccurredAt       time.Time      `json:"occurred_at" gorm:"index;not null"`
	CreatedAt        time.Time      `json:"created_at"`
}
