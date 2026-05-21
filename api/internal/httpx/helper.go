package httpx

import "net/http"

func BadRequest(err error) *AppError {
	return &AppError{Status: http.StatusBadRequest, Message: err.Error(), Err: err}
}

func Unauthorized(err error) *AppError {
	return &AppError{Status: http.StatusUnauthorized, Message: err.Error(), Err: err}
}

func Conflict(err error) *AppError {
	return &AppError{Status: http.StatusConflict, Message: err.Error(), Err: err}
}

func NotFound(err error) *AppError {
	return &AppError{Status: http.StatusNotFound, Message: err.Error(), Err: err}
}

func Internal(err error) *AppError {
	return &AppError{Status: http.StatusInternalServerError, Message: "erro interno", Err: err}
}

func MethodNotAllowed(err error) *AppError {
	return &AppError{Status: http.StatusMethodNotAllowed, Message: "method not allowed", Err: err}
}
