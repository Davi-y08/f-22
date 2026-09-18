export type CameraStatus = "online" | "offline" | "unknown" | string;

export type UserProfile = {
  id: string;
  name: string;
  email: string;
  role: string;
};

export type LoginResponse = {
  access_token: string;
  message: string;
  user: UserProfile;
};

export type Camera = {
  id: string;
  name: string;
  location: string;
  url: string;
  status: CameraStatus;
  agent_id?: string;
  external_id?: string;
  last_seen_at?: string;
  created_at?: string;
  updated_at?: string;
};

export type DetectionEvent = {
  id: string;
  agent_id: string;
  camera_id?: string;
  camera_external_id: string;
  camera_name: string;
  event_type: string;
  confidence: number;
  model_alias: string;
  label: string;
  bbox?: string;
  frame_size?: string;
  zone?: string;
  snapshot_path?: string;
  public_snapshot_url?: string;
  snapshot_url?: string;
  snapshot_mime_type?: string;
  snapshot_filename?: string;
  snapshot_size?: number;
  metadata?: string;
  occurred_at: string;
  created_at: string;
};

export type NotificationDevice = {
  id: string;
  expo_push_token: string;
  platform: string;
  device_name?: string;
  enabled: boolean;
  last_seen_at: string;
};
