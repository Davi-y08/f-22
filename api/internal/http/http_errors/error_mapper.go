package httperrors

import (
	"errors"

	shared "stealth-lens/internal/application"
	domainAgent "stealth-lens/internal/domain/agent"
	domainCamera "stealth-lens/internal/domain/camera"
	domainDetection "stealth-lens/internal/domain/detection"
	domainUser "stealth-lens/internal/domain/user"
	"stealth-lens/internal/httpx"
)

func MapErrorsUser(err error) *httpx.AppError {
	switch {
	case errors.Is(err, domainUser.ErrUserInvalidData):
		return httpx.BadRequest(err)
	case errors.Is(err, domainUser.ErrPasswordDontMatch):
		return httpx.BadRequest(err)
	case errors.Is(err, domainUser.ErrUserNotFound):
		return httpx.NotFound(err)
	case errors.Is(err, domainUser.ErrExistingUser):
		return httpx.Conflict(err)
	case errors.Is(err, domainUser.ErrInvalidCredentials):
		return httpx.Unauthorized(err)
	case errors.Is(err, domainUser.ErrHashedPassword), errors.Is(err, shared.ErrInDataBase):
		return httpx.Internal(err)
	default:
		return httpx.Internal(errors.New("erro interno -> user"))
	}
}

func MapErrorsCamera(err error) *httpx.AppError {
	switch {
	case errors.Is(err, domainCamera.ErrCameraInvalidData):
		return httpx.BadRequest(err)
	case errors.Is(err, domainCamera.ErrCameraNotFound):
		return httpx.NotFound(err)
	case errors.Is(err, shared.ErrInDataBase):
		return httpx.Internal(err)
	default:
		return httpx.Internal(errors.New("erro interno -> camera"))
	}
}

func MapErrorsAgentAccessKey(err error) *httpx.AppError {
	switch {
	case errors.Is(err, domainAgent.ErrAgentAccessKeyInvalidData):
		return httpx.BadRequest(err)
	case errors.Is(err, domainAgent.ErrAgentAccessKeyInvalid):
		return httpx.Unauthorized(err)
	case errors.Is(err, domainAgent.ErrAgentAccessKeyNotFound):
		return httpx.NotFound(err)
	case errors.Is(err, shared.ErrInDataBase):
		return httpx.Internal(err)
	default:
		return httpx.Internal(errors.New("erro interno -> agent access key"))
	}
}

func MapErrorsDetectionEvent(err error) *httpx.AppError {
	switch {
	case errors.Is(err, domainDetection.ErrDetectionEventInvalidData):
		return httpx.BadRequest(err)
	case errors.Is(err, shared.ErrInDataBase):
		return httpx.Internal(err)
	default:
		return httpx.Internal(errors.New("erro interno -> detection event"))
	}
}
