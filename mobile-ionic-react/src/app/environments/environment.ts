export const DEFAULT_API_BASE_URL = "https://api-f22.onrender.com";

function normalizeApiBaseUrl(value?: string) {
  const candidate = value?.trim().replace(/\/+$/, "");

  if (!candidate) {
    return DEFAULT_API_BASE_URL;
  }

  try {
    const url = new URL(candidate);
    const isHttpUrl = url.protocol === "http:" || url.protocol === "https:";

    if (!isHttpUrl) {
      return DEFAULT_API_BASE_URL;
    }

    return url.toString().replace(/\/$/, "");
  } catch {
    return DEFAULT_API_BASE_URL;
  }
}

export const API_BASE_URL = normalizeApiBaseUrl(
  import.meta.env.VITE_API_BASE_URL as string | undefined,
);
