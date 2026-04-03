package user

import "errors"

var (
	ErrUserNotFound 		= 	errors.New("usuário não encontrado") // not found
	ErrUserInvalidData 		= 	errors.New("dados inválidos") // bad request
	ErrPasswordDontMatch	= 	errors.New("senhas não coincidem") // bad request
	ErrHashedPassword 		= 	errors.New("erro ao hashear senha") // internal error
	ErrExistingUser 		= 	errors.New("usuário já existente") // conflict error
	ErrInvalidCredentials 	= 	errors.New("credenciais inválidas") // unauthorized error
)