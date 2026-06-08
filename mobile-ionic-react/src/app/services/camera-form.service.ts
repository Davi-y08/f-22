import { DEFAULT_API_BASE_URL } from "../environments/environment";
import type { CameraPayload, CameraStatus } from "../models/camera.model";

export const emptyCameraForm: CameraPayload = {
  location: "",
  name: "",
  status: "unknown",
  url: "",
};

const validCameraStatuses: CameraStatus[] = ["unknown", "online", "offline"];
const allowedCameraProtocols = ["rtsp:", "rtsps:", "http:", "https:", "local:"];

function isValidIPv4(value: string) {
  const parts = value.split(".");

  if (parts.length !== 4) {
    return false;
  }

  return parts.every((part) => {
    if (!/^\d{1,3}$/.test(part)) return false;
    const number = Number(part);
    return number >= 0 && number <= 255;
  });
}

export function isValidCameraSource(value: string) {
  const source = value.trim();

  if (!source) return false;
  if (source === "0" || source.toLowerCase().startsWith("local://")) return true;
  if (isValidIPv4(source)) return true;

  try {
    const parsed = new URL(source);
    return Boolean(parsed.hostname) && allowedCameraProtocols.includes(parsed.protocol);
  } catch {
    return false;
  }
}

export function validateCameraForm(payload: CameraPayload) {
  const errors: string[] = [];

  if (!payload.name.trim()) {
    errors.push("Informe o nome da câmera.");
  }

  if (!payload.location.trim()) {
    errors.push("Informe o local da câmera.");
  }

  if (!validCameraStatuses.includes(payload.status)) {
    errors.push("Selecione um status válido.");
  }

  return errors;
}

export function normalizeCameraPayload(payload: CameraPayload): CameraPayload {
  const trimmedUrl = payload.url.trim();

  return {
    location: payload.location.trim(),
    name: payload.name.trim(),
    status: payload.status,
    url: isValidCameraSource(trimmedUrl) ? trimmedUrl : DEFAULT_API_BASE_URL,
  };
}

export function cameraUsesDefaultUrl(payload: CameraPayload, normalized: CameraPayload) {
  return normalized.url === DEFAULT_API_BASE_URL && !isValidCameraSource(payload.url);
}
