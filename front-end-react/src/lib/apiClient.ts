import { API_BASE_URL, DEFAULT_API_BASE_URL } from "../config/api";
import type { Camera, CameraPayload } from "../types/camera";

export type ApiMethod = "GET" | "POST" | "PUT" | "DELETE";

export type ApiRequestOptions = {
  body?: unknown;
  method?: ApiMethod;
};

export type UserProfile = {
  id: string;
  name: string;
  email: string;
  role: string;
};

export type AgentAccessKey = {
  access_key?: string;
  created_at?: string;
  id: string;
  key_prefix: string;
  last_used_at?: string;
  name: string;
  revoked_at?: string;
};

export type DetectionEvent = {
  agent_id: string;
  bbox?: string;
  camera_external_id: string;
  camera_id?: string;
  camera_name: string;
  confidence: number;
  created_at: string;
  event_type: string;
  frame_size?: string;
  id: string;
  label: string;
  local_event_id?: string;
  metadata?: string;
  model_alias: string;
  occurred_at: string;
  snapshot_filename?: string;
  snapshot_mime_type?: string;
  snapshot_path?: string;
  snapshot_size?: number;
  snapshot_url?: string;
  zone?: string;
};

export type LoginPayload = {
  email: string;
  password: string;
};

export type SignupPayload = {
  name: string;
  email: string;
  password: string;
  confirm_password: string;
};

export class ApiClientError extends Error {
  payload: unknown;
  status: number;

  constructor(message: string, status: number, payload: unknown) {
    super(message);
    this.name = "ApiClientError";
    this.status = status;
    this.payload = payload;
  }
}

export async function apiRequest<T>(
  path: string,
  options: ApiRequestOptions = {},
): Promise<T> {
  try {
    return await requestFromApi<T>(API_BASE_URL, path, options);
  } catch (error) {
    const shouldRetryWithDefault =
      error instanceof TypeError && API_BASE_URL !== DEFAULT_API_BASE_URL;

    if (!shouldRetryWithDefault) {
      throw error;
    }

    return requestFromApi<T>(DEFAULT_API_BASE_URL, path, options);
  }
}

async function requestFromApi<T>(
  apiBaseUrl: string,
  path: string,
  options: ApiRequestOptions = {},
): Promise<T> {
  const headers: Record<string, string> = {
    Accept: "application/json",
  };

  if (options.body !== undefined) {
    headers["Content-Type"] = "application/json";
  }

  const response = await fetch(`${apiBaseUrl}${path}`, {
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
    credentials: "include",
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

export function apiAssetUrl(pathOrUrl: string, apiBaseUrl = API_BASE_URL) {
  if (pathOrUrl.startsWith("http://") || pathOrUrl.startsWith("https://")) {
    return pathOrUrl;
  }

  return `${apiBaseUrl}${pathOrUrl.startsWith("/") ? pathOrUrl : `/${pathOrUrl}`}`;
}

export async function fetchApiAssetBlob(pathOrUrl: string): Promise<Blob> {
  try {
    return await fetchAssetBlobFromApi(API_BASE_URL, pathOrUrl);
  } catch (error) {
    const shouldRetryWithDefault =
      error instanceof TypeError &&
      API_BASE_URL !== DEFAULT_API_BASE_URL &&
      !pathOrUrl.startsWith("http://") &&
      !pathOrUrl.startsWith("https://");

    if (!shouldRetryWithDefault) {
      throw error;
    }

    return fetchAssetBlobFromApi(DEFAULT_API_BASE_URL, pathOrUrl);
  }
}

async function fetchAssetBlobFromApi(apiBaseUrl: string, pathOrUrl: string): Promise<Blob> {
  const response = await fetch(apiAssetUrl(pathOrUrl, apiBaseUrl), {
    credentials: "include",
    headers: {
      Accept: "image/*",
    },
    method: "GET",
  });

  if (!response.ok) {
    const payload = await response.text().catch(() => "");
    throw new ApiClientError(
      payload.trim() || `HTTP ${response.status}`,
      response.status,
      payload,
    );
  }

  return response.blob();
}

export function getErrorMessage(error: unknown) {
  if (error instanceof ApiClientError) {
    return error.message;
  }

  if (error instanceof TypeError) {
    return "Não foi possível conectar com a API. Verifique se o backend está rodando.";
  }

  if (error instanceof Error) {
    return error.message;
  }

  return "Algo inesperado aconteceu.";
}

export const authApi = {
  login(payload: LoginPayload) {
    return apiRequest<UserProfile>("/users/login", {
      body: payload,
      method: "POST",
    });
  },
  me() {
    return apiRequest<UserProfile>("/users/me");
  },
  signup(payload: SignupPayload) {
    return apiRequest<UserProfile>("/users", {
      body: payload,
      method: "POST",
    });
  },
};

export const cameraApi = {
  create(payload: CameraPayload) {
    return apiRequest<Camera>("/cameras", {
      body: payload,
      method: "POST",
    });
  },
  delete(id: string) {
    return apiRequest<void>(`/cameras/${id}`, {
      method: "DELETE",
    });
  },
  list() {
    return apiRequest<Camera[]>("/cameras");
  },
  update(id: string, payload: CameraPayload) {
    return apiRequest<Camera>(`/cameras/${id}`, {
      body: payload,
      method: "PUT",
    });
  },
};

export const agentKeyApi = {
  create(name: string) {
    return apiRequest<AgentAccessKey>("/agent-keys", {
      body: { name },
      method: "POST",
    });
  },
  list() {
    return apiRequest<AgentAccessKey[]>("/agent-keys");
  },
  revoke(id: string) {
    return apiRequest<void>(`/agent-keys/${id}`, {
      method: "DELETE",
    });
  },
};

export const detectionEventApi = {
  list(limit = 80) {
    return apiRequest<DetectionEvent[]>(`/events?limit=${limit}`);
  },
};
