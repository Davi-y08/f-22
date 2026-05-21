package security

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"strings"
)

const agentAccessKeyPrefix = "slk_"

func GenerateAgentAccessKey() (string, error) {
	randomBytes := make([]byte, 32)
	if _, err := rand.Read(randomBytes); err != nil {
		return "", err
	}

	return agentAccessKeyPrefix + base64.RawURLEncoding.EncodeToString(randomBytes), nil
}

func HashAgentAccessKey(value string) string {
	sum := sha256.Sum256([]byte(strings.TrimSpace(value)))
	return hex.EncodeToString(sum[:])
}

func AgentAccessKeyPrefix(value string) string {
	trimmed := strings.TrimSpace(value)
	if len(trimmed) <= 16 {
		return trimmed
	}

	return trimmed[:16]
}

func IsValidAgentAccessKeyFormat(value string) bool {
	trimmed := strings.TrimSpace(value)
	return strings.HasPrefix(trimmed, agentAccessKeyPrefix) && len(trimmed) >= 24
}
