const fallbackApiBaseUrl = "https://api-f22.onrender.com";

export const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim().replace(/\/$/, "") ||
  fallbackApiBaseUrl;
