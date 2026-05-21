package camera

import (
	"context"
	"errors"
	"strings"

	"github.com/google/uuid"
	"gorm.io/gorm"

	shared "stealth-lens/internal/application"
	domainCamera "stealth-lens/internal/domain/camera"
	repo "stealth-lens/internal/repository"
)

type CameraService struct {
	repo *repo.CameraRepository
}

func NewCameraService(repo *repo.CameraRepository) *CameraService {
	return &CameraService{repo: repo}
}

func (s *CameraService) CreateCameraService(ctx context.Context, dto CreateCameraDto) (*domainCamera.Camera, error) {
	newCamera, err := ValidateCreateDto(dto)
	if err != nil {
		return nil, err
	}

	if err := s.repo.CreateCamera(ctx, newCamera); err != nil {
		return nil, shared.ErrInDataBase
	}

	return newCamera, nil
}

func (s *CameraService) SyncAgentCamerasService(ctx context.Context, userID uuid.UUID, dto SyncAgentCamerasDto) (*SyncAgentCamerasResult, error) {
	agentID := strings.TrimSpace(dto.AgentID)
	if agentID == "" || len(dto.Cameras) == 0 || len(dto.Cameras) > 500 {
		return nil, domainCamera.ErrCameraInvalidData
	}

	result := &SyncAgentCamerasResult{
		AgentID: agentID,
		Cameras: make([]SyncedCameraView, 0, len(dto.Cameras)),
	}

	for _, item := range dto.Cameras {
		incomingCamera, err := ValidateAgentCameraDto(agentID, userID, item)
		if err != nil {
			return nil, err
		}

		syncedCamera, created, err := s.repo.UpsertCameraFromAgent(ctx, userID, incomingCamera)
		if err != nil {
			return nil, shared.ErrInDataBase
		}

		if created {
			result.Created++
		} else {
			result.Updated++
		}

		result.Cameras = append(result.Cameras, SyncedCameraView{
			ID:         syncedCamera.ID,
			ExternalID: syncedCamera.ExternalID,
			Name:       syncedCamera.Name,
			URL:        syncedCamera.URL,
			Status:     syncedCamera.Status,
			Created:    created,
		})
	}

	return result, nil
}

func (s *CameraService) ListCamerasService(ctx context.Context, userID uuid.UUID) ([]domainCamera.Camera, error) {
	cameras, err := s.repo.GetCamerasByUser(ctx, userID)
	if err != nil {
		return nil, shared.ErrInDataBase
	}

	return cameras, nil
}

func (s *CameraService) GetCameraByIDService(ctx context.Context, cameraID, userID uuid.UUID) (*domainCamera.Camera, error) {
	foundCamera, err := s.repo.GetCameraByID(ctx, cameraID, userID)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, domainCamera.ErrCameraNotFound
		}

		return nil, shared.ErrInDataBase
	}

	return foundCamera, nil
}

func (s *CameraService) UpdateCameraService(ctx context.Context, cameraID, userID uuid.UUID, dto UpdateCameraDto) (*domainCamera.Camera, error) {
	cameraChanges, err := ValidateUpdateDto(dto)
	if err != nil {
		return nil, err
	}

	updatedCamera, err := s.repo.UpdateCamera(ctx, cameraID, userID, cameraChanges)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, domainCamera.ErrCameraNotFound
		}

		return nil, shared.ErrInDataBase
	}

	return updatedCamera, nil
}

func (s *CameraService) DeleteCameraService(ctx context.Context, cameraID, userID uuid.UUID) error {
	if err := s.repo.DeleteCamera(ctx, cameraID, userID); err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return domainCamera.ErrCameraNotFound
		}

		return shared.ErrInDataBase
	}

	return nil
}
