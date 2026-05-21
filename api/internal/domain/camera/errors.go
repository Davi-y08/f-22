package camera

import "errors"

var (
	ErrCameraInvalidData = errors.New("dados da camera invalidos")
	ErrCameraNotFound    = errors.New("camera nao encontrada")
)
