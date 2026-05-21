package repository

import (
	"context"

	"github.com/google/uuid"
	"gorm.io/gorm"

	"stealth-lens/internal/domain/detection"
)

type DetectionEventRepository struct {
	db *gorm.DB
}

func NewDetectionEventRepository(db *gorm.DB) *DetectionEventRepository {
	return &DetectionEventRepository{db: db}
}

func (r *DetectionEventRepository) CreateDetectionEvent(ctx context.Context, event *detection.DetectionEvent) error {
	return r.db.WithContext(ctx).Model(&detection.DetectionEvent{}).Create(event).Error
}

func (r *DetectionEventRepository) GetDetectionEventsByUser(ctx context.Context, userID uuid.UUID, limit int) ([]detection.DetectionEvent, error) {
	var events []detection.DetectionEvent
	if err := r.db.WithContext(ctx).Model(&detection.DetectionEvent{}).
		Where("user_id = ?", userID).
		Order("occurred_at DESC").
		Limit(limit).
		Find(&events).Error; err != nil {
		return nil, err
	}

	return events, nil
}
