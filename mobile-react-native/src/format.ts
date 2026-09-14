import type { CameraStatus, DetectionEvent } from "./types";

const dateTimeFormatter = new Intl.DateTimeFormat("pt-BR", {
  dateStyle: "short",
  timeStyle: "medium",
});

const timeFormatter = new Intl.DateTimeFormat("pt-BR", {
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
});

export function formatDateTime(value?: string) {
  const date = parseDate(value);
  return date ? dateTimeFormatter.format(date) : "--";
}

export function formatTime(value?: string) {
  const date = parseDate(value);
  return date ? timeFormatter.format(date) : "--";
}

export function formatConfidence(value?: number) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "--";
  }
  return `${Math.round(value * 100)}%`;
}

export function eventTitle(event: DetectionEvent) {
  const rawLabel = event.label || event.event_type || "alerta";
  const normalized = rawLabel.trim().toLowerCase();
  const knownLabels: Record<string, string> = {
    fire: "Fogo",
    knife: "Faca",
    person: "Pessoa",
    smoke: "Fumaça",
    smoking: "Pessoa fumando",
  };

  return knownLabels[normalized] ?? capitalize(rawLabel.replace(/[_-]+/g, " "));
}

export function cameraStatusLabel(status?: CameraStatus) {
  const normalized = String(status ?? "unknown").toLowerCase();
  if (normalized === "online") {
    return "Online";
  }
  if (normalized === "offline") {
    return "Offline";
  }
  return "Sem sinal";
}

export function cameraStatusTone(status?: CameraStatus) {
  const normalized = String(status ?? "unknown").toLowerCase();
  if (normalized === "online") {
    return "ok";
  }
  if (normalized === "offline") {
    return "danger";
  }
  return "muted";
}

function parseDate(value?: string) {
  if (!value) {
    return null;
  }

  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

function capitalize(value: string) {
  const clean = value.trim();
  if (!clean) {
    return "Alerta";
  }
  return clean.charAt(0).toUpperCase() + clean.slice(1);
}
