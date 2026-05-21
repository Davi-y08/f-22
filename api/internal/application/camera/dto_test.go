package camera

import (
	"testing"

	"github.com/google/uuid"
)

func TestValidateCreateDtoAcceptsIPRTSPAndLocalURL(t *testing.T) {
	userID := uuid.New()
	cases := []CreateCameraDto{
		{Name: "Entrada", Location: "Portao", URL: "192.168.1.10", UserID: userID},
		{Name: "Garagem", Location: "Fundos", URL: "rtsp://192.168.1.11:554/stream", Status: "online", UserID: userID},
		{Name: "Webcam", Location: "Agente local", URL: "local://0", Status: "unknown", UserID: userID},
	}

	for _, testCase := range cases {
		if _, err := ValidateCreateDto(testCase); err != nil {
			t.Fatalf("esperava camera valida para %q, recebeu erro: %v", testCase.URL, err)
		}
	}
}

func TestValidateCreateDtoRejectsInvalidSourceAndStatus(t *testing.T) {
	cases := []CreateCameraDto{
		{Name: "Entrada", Location: "Portao", URL: "arquivo-local.mp4", UserID: uuid.New()},
		{Name: "Entrada", Location: "Portao", URL: "192.168.1.10", Status: "falhando", UserID: uuid.New()},
	}

	for _, testCase := range cases {
		if _, err := ValidateCreateDto(testCase); err == nil {
			t.Fatalf("esperava camera invalida para %+v", testCase)
		}
	}
}

func TestValidateAgentCameraDtoAcceptsDiscoveredCamera(t *testing.T) {
	camera, err := ValidateAgentCameraDto("agent-1", uuid.New(), AgentCameraDto{
		ExternalID: "local-0",
		Name:       "Webcam Local 0",
		URL:        "local://0",
		Status:     "online",
	})
	if err != nil {
		t.Fatalf("esperava camera do agente valida: %v", err)
	}
	if camera.Location == "" || camera.AgentID != "agent-1" || camera.ExternalID != "local-0" {
		t.Fatalf("camera normalizada incorretamente: %+v", camera)
	}
}
