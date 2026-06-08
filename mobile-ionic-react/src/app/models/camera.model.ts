export type CameraStatus = "online" | "offline" | "unknown";

export type Camera = {
  agent_id?: string;
  created_at?: string;
  external_id?: string;
  id: string;
  last_seen_at?: string;
  location: string;
  name: string;
  status: CameraStatus | string;
  updated_at?: string;
  url: string;
};

export type CameraPayload = {
  location: string;
  name: string;
  status: CameraStatus;
  url: string;
};
