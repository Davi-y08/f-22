import Constants from "expo-constants";

import type {
  Camera,
  DetectionEvent,
  LoginResponse,
  NotificationDevice,
  UserProfile,
} from "./types";

const DEFAULT_API_BASE_URL = "https://api-f22.onrender.com";

type RequestOptions = {
  body?: unknown;
  method?: "GET" | "POST" | "PUT" | "DELETE";
  token?: string | null;
};

export class ApiClientError extends Error {
  status: number;
  payload: unknown;

  constructor(message: string, status: number, payload: unknown) {
    super(message);
    this.name = "ApiClientError";
    this.status = status;
    this.payload = payload;
  }
}

function configuredApiBaseUrl() {
  const extra = Constants.expoConfig?.extra as { apiBaseUrl?: string } | undefined;
  return normalizeApiBaseUrl(extra?.apiBaseUrl);
}

function normalizeApiBaseUrl(value?: string) {
  const candidate = value?.trim().replace(/\/+$/, "");
  if (!candidate) {
    return DEFAULT_API_BASE_URL;
  }

  try {
    const url = new URL(candidate);
    if (url.protocol !== "http:" && url.protocol !== "https:") {
      return DEFAULT_API_BASE_URL;
    }
    return url.toString().replace(/\/$/, "");
  } catch {
    return DEFAULT_API_BASE_URL;
  }
}

export const API_BASE_URL = configuredApiBaseUrl();

export function snapshotUrl(pathOrUrl?: string) {
  if (!pathOrUrl) {
    return "";
  }
  if (pathOrUrl.startsWith("http://") || pathOrUrl.startsWith("https://")) {
    return pathOrUrl;
  }
  return `${API_BASE_URL}${pathOrUrl.startsWith("/") ? pathOrUrl : `/${pathOrUrl}`}`;
}

export async function apiRequest<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const headers: Record<string, string> = {
    Accept: "application/json",
  };

  if (options.body !== undefined) {
    headers["Content-Type"] = "application/json";
  }
  if (options.token) {
    headers.Authorization = `Bearer ${options.token}`;
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
    headers,
    method: options.method ?? "GET",
  });

  const contentType = response.headers.get("content-type") ?? "";
  const payload =
    response.status === 204
      ? null
      : contentType.includes("application/json")
        ? await response.json()
        : await response.text();

  if (!response.ok) {
    const message =
      typeof payload === "object" && payload !== null && "error" in payload
        ? String((payload as { error: unknown }).error)
        : `HTTP ${response.status}`;

    throw new ApiClientError(message, response.status, payload);
  }

  return payload as T;
}

export function getErrorMessage(error: unknown) {
  if (error instanceof ApiClientError) {
    if (error.status === 401) {
      return "Sessão expirada. Faça login novamente.";
    }
    return error.message;
  }

  if (error instanceof TypeError) {
    return "Não foi possível conectar com a API.";
  }

  if (error instanceof Error) {
    return error.message;
  }

  return "Algo inesperado aconteceu.";
}

export const authApi = {
  login(email: string, password: string) {
    return apiRequest<LoginResponse>("/users/login", {
      body: { email, password },
      method: "POST",
    });
  },
  me(token: string) {
    return apiRequest<UserProfile>("/users/me", { token });
  },
};

export const cameraApi = {
  list(token: string) {
    return apiRequest<Camera[]>("/cameras", { token });
  },
};

export const eventApi = {
  list(token: string, limit = 50) {
    return apiRequest<DetectionEvent[]>(`/events?limit=${limit}`, { token });
  },
};

export const notificationApi = {
  registerDevice(
    token: string,
    payload: { device_name?: string; expo_push_token: string; platform: string },
  ) {
    return apiRequest<NotificationDevice>("/notification-devices", {
      body: payload,
      method: "POST",
      token,
    });
  },
};
