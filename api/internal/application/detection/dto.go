package detection

import (
	"encoding/json"
	"strings"
	"time"

	"github.com/google/uuid"

	domainDetection "stealth-lens/internal/domain/detection"
)

type AgentDetectionEventDto struct {
	EventID      string         `json:"event_id"`
	AgentID      string         `json:"agent_id"`
	Timestamp    string         `json:"timestamp"`
	CameraID     string         `json:"camera_id"`
	CameraName   string         `json:"camera_name"`
	EventType    string         `json:"event_type"`
	Confidence   float64        `json:"confidence"`
	ModelAlias   string         `json:"model_alias"`
	Label        string         `json:"label"`
	BBox         []int          `json:"bbox"`
	Zone         *string        `json:"zone"`
	SnapshotPath *string        `json:"snapshot_path"`
	FrameSize    []int          `json:"frame_size"`
	Metadata     map[string]any `json:"metadata"`
}

func ValidateAgentDetectionEventDto(userID uuid.UUID, dto AgentDetectionEventDto) (*domainDetection.DetectionEvent, error) {
	agentID := strings.TrimSpace(dto.AgentID)
	eventType := strings.TrimSpace(strings.ToLower(dto.EventType))
	cameraExternalID := strings.TrimSpace(dto.CameraID)

	if userID == uuid.Nil || agentID == "" || eventType == "" || cameraExternalID == "" {
		return nil, domainDetection.ErrDetectionEventInvalidData
	}

	occurredAt := time.Now().UTC()
	if strings.TrimSpace(dto.Timestamp) != "" {
		parsed, err := time.Parse(time.RFC3339, dto.Timestamp)
		if err != nil {
			return nil, domainDetection.ErrDetectionEventInvalidData
		}
		occurredAt = parsed.UTC()
	}

	bbox, err := json.Marshal(dto.BBox)
	if err != nil {
		return nil, domainDetection.ErrDetectionEventInvalidData
	}

	frameSize, err := json.Marshal(dto.FrameSize)
	if err != nil {
		return nil, domainDetection.ErrDetectionEventInvalidData
	}

	metadata := dto.Metadata
	if metadata == nil {
		metadata = map[string]any{}
	}
	metadataBytes, err := json.Marshal(metadata)
	if err != nil {
		return nil, domainDetection.ErrDetectionEventInvalidData
	}

	return &domainDetection.DetectionEvent{
		UserID:           userID,
		AgentID:          agentID,
		LocalEventID:     strings.TrimSpace(dto.EventID),
		CameraExternalID: cameraExternalID,
		CameraName:       strings.TrimSpace(dto.CameraName),
		EventType:        eventType,
		Confidence:       dto.Confidence,
		ModelAlias:       strings.TrimSpace(dto.ModelAlias),
		Label:            strings.TrimSpace(dto.Label),
		BBox:             string(bbox),
		FrameSize:        string(frameSize),
		Zone:             cleanOptionalString(dto.Zone),
		SnapshotPath:     cleanOptionalString(dto.SnapshotPath),
		Metadata:         string(metadataBytes),
		OccurredAt:       occurredAt,
	}, nil
}

func cleanOptionalString(value *string) *string {
	if value == nil {
		return nil
	}
	cleaned := strings.TrimSpace(*value)
	if cleaned == "" {
		return nil
	}
	return &cleaned
}
