export type CameraStatus = "online" | "offline" | "unknown";

export type Camera = {
  id: string;
  name: string;
  location: string;
  url: string;
  status: CameraStatus | string;
  agent_id?: string;
  external_id?: string;
  last_seen_at?: string;
  created_at?: string;
  updated_at?: string;
};

export type CameraPayload = {
  name: string;
  location: string;
  url: string;
  status: CameraStatus;
};
