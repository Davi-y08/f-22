package agent

import "errors"

var (
	ErrAgentAccessKeyInvalidData = errors.New("dados da chave de agente invalidos")
	ErrAgentAccessKeyInvalid     = errors.New("chave de agente invalida")
	ErrAgentAccessKeyNotFound    = errors.New("chave de agente nao encontrada")
)
