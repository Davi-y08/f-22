package security

import "testing"

func TestGenerateAgentAccessKey(t *testing.T) {
	key, err := GenerateAgentAccessKey()
	if err != nil {
		t.Fatalf("erro ao gerar chave: %v", err)
	}

	if !IsValidAgentAccessKeyFormat(key) {
		t.Fatalf("formato de chave invalido: %s", key)
	}

	if HashAgentAccessKey(key) == HashAgentAccessKey(key+"x") {
		t.Fatal("hash da chave nao pode colidir para valores diferentes")
	}
}
