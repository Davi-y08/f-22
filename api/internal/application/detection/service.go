package detection

import (
	"context"
	"errors"

	"github.com/google/uuid"
	"gorm.io/gorm"

	shared "stealth-lens/internal/application"
	domainDetection "stealth-lens/internal/domain/detection"
	repo "stealth-lens/internal/repository"
)

type DetectionEventService struct {
	repo       *repo.DetectionEventRepository
	cameraRepo *repo.CameraRepository
}

func NewDetectionEventService(repo *repo.DetectionEventRepository, cameraRepo *repo.CameraRepository) *DetectionEventService {
	return &DetectionEventService{repo: repo, cameraRepo: cameraRepo}
}

func (s *DetectionEventService) CreateFromAgent(ctx context.Context, userID uuid.UUID, dto AgentDetectionEventDto) (*domainDetection.DetectionEvent, error) {
	event, err := ValidateAgentDetectionEventDto(userID, dto)
	if err != nil {
		return nil, err
	}

	camera, err := s.cameraRepo.GetCameraByAgentExternalID(ctx, userID, event.AgentID, event.CameraExternalID)
	if err == nil && camera != nil {
		event.CameraID = &camera.ID
	} else if err != nil && !errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, shared.ErrInDataBase
	}

	if err := s.repo.CreateDetectionEvent(ctx, event); err != nil {
		return nil, shared.ErrInDataBase
	}

	return event, nil
}

func (s *DetectionEventService) ListByUser(ctx context.Context, userID uuid.UUID, limit int) ([]domainDetection.DetectionEvent, error) {
	if limit <= 0 || limit > 200 {
		limit = 100
	}

	events, err := s.repo.GetDetectionEventsByUser(ctx, userID, limit)
	if err != nil {
		return nil, shared.ErrInDataBase
	}

	return events, nil
}
