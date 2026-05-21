package repository

import (
	"context"
	"errors"

	"github.com/google/uuid"
	"gorm.io/gorm"

	"stealth-lens/internal/domain/camera"
)

type CameraRepository struct {
	db *gorm.DB
}

func NewCameraRepository(db *gorm.DB) *CameraRepository {
	return &CameraRepository{db: db}
}

func (r *CameraRepository) CreateCamera(ctx context.Context, foundCamera *camera.Camera) error {
	return r.db.WithContext(ctx).Model(&camera.Camera{}).Create(foundCamera).Error
}

func (r *CameraRepository) UpsertCameraFromAgent(ctx context.Context, userID uuid.UUID, incomingCamera *camera.Camera) (*camera.Camera, bool, error) {
	foundCamera, err := r.FindCameraForAgentSync(ctx, userID, incomingCamera.AgentID, incomingCamera.ExternalID, incomingCamera.URL)
	if err != nil && !errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, false, err
	}

	if errors.Is(err, gorm.ErrRecordNotFound) {
		if err := r.CreateCamera(ctx, incomingCamera); err != nil {
			return nil, false, err
		}
		return incomingCamera, true, nil
	}

	foundCamera.Name = incomingCamera.Name
	foundCamera.Location = incomingCamera.Location
	foundCamera.URL = incomingCamera.URL
	foundCamera.Status = incomingCamera.Status
	foundCamera.AgentID = incomingCamera.AgentID
	foundCamera.ExternalID = incomingCamera.ExternalID
	foundCamera.LastSeenAt = incomingCamera.LastSeenAt

	if err := r.db.WithContext(ctx).Save(foundCamera).Error; err != nil {
		return nil, false, err
	}

	return foundCamera, false, nil
}

func (r *CameraRepository) FindCameraForAgentSync(ctx context.Context, userID uuid.UUID, agentID, externalID, cameraURL string) (*camera.Camera, error) {
	var foundCamera camera.Camera
	query := r.db.WithContext(ctx).Model(&camera.Camera{}).Where("user_id = ?", userID)

	if agentID != "" && externalID != "" {
		if err := query.Where("agent_id = ? AND external_id = ?", agentID, externalID).First(&foundCamera).Error; err == nil {
			return &foundCamera, nil
		} else if !errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, err
		}
	}

	if cameraURL != "" {
		if err := r.db.WithContext(ctx).Model(&camera.Camera{}).Where("user_id = ? AND url = ?", userID, cameraURL).First(&foundCamera).Error; err == nil {
			return &foundCamera, nil
		} else if !errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, err
		}
	}

	return nil, gorm.ErrRecordNotFound
}

func (r *CameraRepository) GetCameraByAgentExternalID(ctx context.Context, userID uuid.UUID, agentID, externalID string) (*camera.Camera, error) {
	var foundCamera camera.Camera
	if err := r.db.WithContext(ctx).Model(&camera.Camera{}).
		Where("user_id = ? AND agent_id = ? AND external_id = ?", userID, agentID, externalID).
		First(&foundCamera).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, gorm.ErrRecordNotFound
		}
		return nil, err
	}

	return &foundCamera, nil
}

func (r *CameraRepository) GetCamerasByUser(ctx context.Context, userID uuid.UUID) ([]camera.Camera, error) {
	var cameras []camera.Camera
	if err := r.db.WithContext(ctx).Model(&camera.Camera{}).Where("user_id = ?", userID).Order("created_at DESC").Find(&cameras).Error; err != nil {
		return nil, err
	}

	return cameras, nil
}

func (r *CameraRepository) GetCameraByID(ctx context.Context, cameraID, userID uuid.UUID) (*camera.Camera, error) {
	var foundCamera camera.Camera
	if err := r.db.WithContext(ctx).Model(&camera.Camera{}).Where("id = ? AND user_id = ?", cameraID, userID).First(&foundCamera).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, gorm.ErrRecordNotFound
		}

		return nil, err
	}

	return &foundCamera, nil
}

func (r *CameraRepository) UpdateCamera(ctx context.Context, cameraID, userID uuid.UUID, changes *camera.Camera) (*camera.Camera, error) {
	foundCamera, err := r.GetCameraByID(ctx, cameraID, userID)
	if err != nil {
		return nil, err
	}

	foundCamera.Name = changes.Name
	foundCamera.Location = changes.Location
	foundCamera.URL = changes.URL
	foundCamera.Status = changes.Status

	if err := r.db.WithContext(ctx).Save(foundCamera).Error; err != nil {
		return nil, err
	}

	return foundCamera, nil
}

func (r *CameraRepository) DeleteCamera(ctx context.Context, cameraID, userID uuid.UUID) error {
	result := r.db.WithContext(ctx).Where("id = ? AND user_id = ?", cameraID, userID).Delete(&camera.Camera{})
	if result.Error != nil {
		return result.Error
	}

	if result.RowsAffected == 0 {
		return gorm.ErrRecordNotFound
	}

	return nil
}
