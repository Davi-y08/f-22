package camera

import (
	"net"
	"net/url"
	"strings"
	"time"

	"github.com/google/uuid"

	domainCamera "stealth-lens/internal/domain/camera"
)

type CreateCameraDto struct {
	Name     string    `json:"name"`
	Location string    `json:"location"`
	URL      string    `json:"url"`
	Status   string    `json:"status"`
	UserID   uuid.UUID `json:"-"`
}

type UpdateCameraDto struct {
	Name     string `json:"name"`
	Location string `json:"location"`
	URL      string `json:"url"`
	Status   string `json:"status"`
}

type AgentCameraDto struct {
	ExternalID string `json:"external_id"`
	Name       string `json:"name"`
	Location   string `json:"location"`
	URL        string `json:"url"`
	Source     string `json:"source"`
	Status     string `json:"status"`
}

type SyncAgentCamerasDto struct {
	AgentID string           `json:"agent_id"`
	Cameras []AgentCameraDto `json:"cameras"`
}

type SyncedCameraView struct {
	ID         uuid.UUID `json:"id"`
	ExternalID string    `json:"external_id"`
	Name       string    `json:"name"`
	URL        string    `json:"url"`
	Status     string    `json:"status"`
	Created    bool      `json:"created"`
}

type SyncAgentCamerasResult struct {
	AgentID string             `json:"agent_id"`
	Created int                `json:"created"`
	Updated int                `json:"updated"`
	Cameras []SyncedCameraView `json:"cameras"`
}

func ValidateCreateDto(dto CreateCameraDto) (*domainCamera.Camera, error) {
	status, ok := normalizeStatus(dto.Status)
	if !ok || dto.UserID == uuid.Nil || !hasValidCoreFields(dto.Name, dto.Location, dto.URL) {
		return nil, domainCamera.ErrCameraInvalidData
	}

	return &domainCamera.Camera{
		Name:     strings.TrimSpace(dto.Name),
		Location: strings.TrimSpace(dto.Location),
		URL:      strings.TrimSpace(dto.URL),
		Status:   status,
		UserID:   dto.UserID,
	}, nil
}

func ValidateUpdateDto(dto UpdateCameraDto) (*domainCamera.Camera, error) {
	status, ok := normalizeStatus(dto.Status)
	if !ok || !hasValidCoreFields(dto.Name, dto.Location, dto.URL) {
		return nil, domainCamera.ErrCameraInvalidData
	}

	return &domainCamera.Camera{
		Name:     strings.TrimSpace(dto.Name),
		Location: strings.TrimSpace(dto.Location),
		URL:      strings.TrimSpace(dto.URL),
		Status:   status,
	}, nil
}

func ValidateAgentCameraDto(agentID string, userID uuid.UUID, dto AgentCameraDto) (*domainCamera.Camera, error) {
	status, ok := normalizeStatus(dto.Status)
	if !ok || userID == uuid.Nil {
		return nil, domainCamera.ErrCameraInvalidData
	}

	cameraURL := strings.TrimSpace(dto.URL)
	if cameraURL == "" {
		cameraURL = strings.TrimSpace(dto.Source)
	}

	name := strings.TrimSpace(dto.Name)
	if name == "" {
		name = "Camera descoberta"
	}

	location := strings.TrimSpace(dto.Location)
	if location == "" {
		location = "Agente local"
	}

	externalID := strings.TrimSpace(dto.ExternalID)
	if externalID == "" {
		externalID = cameraURL
	}

	if strings.TrimSpace(agentID) == "" || externalID == "" || !hasValidCoreFields(name, location, cameraURL) {
		return nil, domainCamera.ErrCameraInvalidData
	}

	now := time.Now().UTC()
	return &domainCamera.Camera{
		Name:       name,
		Location:   location,
		URL:        cameraURL,
		Status:     status,
		UserID:     userID,
		AgentID:    strings.TrimSpace(agentID),
		ExternalID: externalID,
		LastSeenAt: &now,
	}, nil
}

func hasValidCoreFields(name, location, source string) bool {
	return strings.TrimSpace(name) != "" && strings.TrimSpace(location) != "" && isValidCameraSource(source)
}

func normalizeStatus(value string) (string, bool) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", domainCamera.StatusUnknown:
		return domainCamera.StatusUnknown, true
	case domainCamera.StatusOnline:
		return domainCamera.StatusOnline, true
	case domainCamera.StatusOffline:
		return domainCamera.StatusOffline, true
	default:
		return "", false
	}
}

func isValidCameraSource(value string) bool {
	source := strings.TrimSpace(value)
	if source == "" {
		return false
	}

	if source == "0" || strings.HasPrefix(strings.ToLower(source), "local://") {
		return true
	}

	if net.ParseIP(source) != nil {
		return true
	}

	parsed, err := url.ParseRequestURI(source)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return false
	}

	switch strings.ToLower(parsed.Scheme) {
	case "rtsp", "rtsps", "http", "https", "local":
		return true
	default:
		return false
	}
}
