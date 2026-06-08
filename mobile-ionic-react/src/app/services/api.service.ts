import { API_BASE_URL, DEFAULT_API_BASE_URL } from "../environments/environment";
import type { AgentAccessKey } from "../models/agent-key.model";
import type { LoginPayload, SignupPayload, UserProfile } from "../models/auth.model";
import type { Camera, CameraPayload } from "../models/camera.model";

export type ApiMethod = "GET" | "POST" | "PUT" | "DELETE";

export type ApiRequestOptions = {
  body?: unknown;
  method?: ApiMethod;
};

export class ApiClientError extends Error {
  payload: unknown;
  status: number;

  constructor(message: string, status: number, payload: unknown) {
    super(message);
    this.name = "ApiClientError";
    this.payload = payload;
    this.status = status;
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
  options: ApiRequestOptions,
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

export function getErrorMessage(error: unknown) {
  if (error instanceof ApiClientError) {
    if (error.status === 401) {
      return "Sua sessão expirou ou não foi encontrada. Faça login novamente.";
    }

    if (
      error.message.toLowerCase().includes("dados da camera invalidos") ||
      error.message.toLowerCase().includes("dados da câmera inválidos")
    ) {
      return "Os dados da câmera estão incompletos. Revise nome, local e status.";
    }

    return error.message;
  }

  if (error instanceof TypeError) {
    return "Não foi possível conectar com a API. Verifique sua internet e tente novamente.";
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
  get(id: string) {
    return apiRequest<Camera>(`/cameras/${id}`);
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

export { API_BASE_URL, DEFAULT_API_BASE_URL };
