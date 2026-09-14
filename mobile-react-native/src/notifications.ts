import Constants from "expo-constants";
import * as Device from "expo-device";
import { Platform } from "react-native";

import { notificationApi } from "./api";
import { eventTitle, formatTime } from "./format";
import type { DetectionEvent } from "./types";

export type NotificationRegistrationState = {
  message: string;
  status: "idle" | "registered" | "denied" | "unavailable" | "failed";
  token?: string;
};

const notificationChannelId = "alerts";
let notificationsModulePromise: Promise<ExpoNotificationsModule | null> | null = null;
let notificationHandlerConfigured = false;

type ExpoNotificationsModule = typeof import("expo-notifications");
type NotificationResponse = import("expo-notifications").NotificationResponse;

export type NotificationListenerCleanup = () => void;

type NotificationListenerOptions = {
  onReceived: () => void;
  onResponse: (eventId: string | null) => void;
};

export async function registerForPushNotifications(
  authToken: string,
): Promise<NotificationRegistrationState> {
  if (Platform.OS === "web") {
    return {
      message: "Notificações push ficam disponíveis no app instalado.",
      status: "unavailable",
    };
  }

  if (isAndroidExpoGo()) {
    return {
      message: "Expo Go no Android não suporta push remoto neste SDK. Use uma development build.",
      status: "unavailable",
    };
  }

  const Notifications = await loadNotificationsModule();
  if (!Notifications) {
    return {
      message: "Módulo de notificações indisponível neste ambiente.",
      status: "unavailable",
    };
  }

  if (Platform.OS === "android") {
    await Notifications.setNotificationChannelAsync(notificationChannelId, {
      importance: Notifications.AndroidImportance.MAX,
      lightColor: "#2DD4BF",
      name: "Alertas de câmera",
      sound: "default",
      vibrationPattern: [0, 250, 250, 250],
    });
  }

  const permission = await ensureNotificationPermission(Notifications);
  if (!permission) {
    return {
      message: "Permissão de notificação negada neste aparelho.",
      status: "denied",
    };
  }

  if (!Device.isDevice) {
    return {
      message: "Push remoto exige aparelho físico; alertas locais seguem ativos.",
      status: "unavailable",
    };
  }

  const projectId = getExpoProjectId();
  if (!projectId) {
    return {
      message: "Configure o EAS projectId para registrar push remoto.",
      status: "failed",
    };
  }

  try {
    const pushToken = await Notifications.getExpoPushTokenAsync({ projectId });
    await notificationApi.registerDevice(authToken, {
      device_name: Device.deviceName ?? Constants.deviceName ?? undefined,
      expo_push_token: pushToken.data,
      platform: Platform.OS,
    });

    return {
      message: "Notificações remotas registradas neste aparelho.",
      status: "registered",
      token: pushToken.data,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : "Falha ao registrar push remoto.";
    return {
      message,
      status: "failed",
    };
  }
}

export async function scheduleLocalAlertNotification(event: DetectionEvent) {
  if (Platform.OS === "web" || isAndroidExpoGo()) {
    return;
  }

  const Notifications = await loadNotificationsModule();
  if (!Notifications) {
    return;
  }

  await Notifications.scheduleNotificationAsync({
    content: {
      body: `${eventTitle(event)} às ${formatTime(event.occurred_at)}`,
      data: {
        camera_name: event.camera_name,
        event_id: event.id,
        snapshot_url: event.snapshot_url ?? "",
      },
      sound: "default",
      title: `Alerta em ${event.camera_name || "câmera"}`,
    },
    trigger: Platform.OS === "android" ? { channelId: notificationChannelId } : null,
  });
}

export async function setupNotificationListeners({
  onReceived,
  onResponse,
}: NotificationListenerOptions): Promise<NotificationListenerCleanup> {
  if (Platform.OS === "web" || isAndroidExpoGo()) {
    return () => undefined;
  }

  const Notifications = await loadNotificationsModule();
  if (!Notifications) {
    return () => undefined;
  }

  const receivedSubscription = Notifications.addNotificationReceivedListener(onReceived);
  const responseSubscription = Notifications.addNotificationResponseReceivedListener((response) => {
    onResponse(eventIdFromNotificationResponse(response));
  });

  const lastResponse = Notifications.getLastNotificationResponse();
  const lastEventId = eventIdFromNotificationResponse(lastResponse);
  if (lastEventId) {
    onResponse(lastEventId);
  }

  return () => {
    receivedSubscription.remove();
    responseSubscription.remove();
  };
}

export function eventIdFromNotificationResponse(
  response: NotificationResponse | null | undefined,
) {
  const data = response?.notification.request.content.data;
  const eventId = data?.event_id ?? data?.eventId;
  return typeof eventId === "string" ? eventId : null;
}

async function ensureNotificationPermission(Notifications: ExpoNotificationsModule) {
  const current = await Notifications.getPermissionsAsync();
  if (current.granted || current.status === Notifications.PermissionStatus.GRANTED) {
    return true;
  }

  const requested = await Notifications.requestPermissionsAsync();
  return requested.granted || requested.status === Notifications.PermissionStatus.GRANTED;
}

function getExpoProjectId() {
  const extra = Constants.expoConfig?.extra as
    | { eas?: { projectId?: string }; projectId?: string }
    | undefined;
  return Constants.easConfig?.projectId ?? extra?.eas?.projectId ?? extra?.projectId ?? null;
}

function isAndroidExpoGo() {
  return Platform.OS === "android" && Constants.expoGoConfig != null;
}

async function loadNotificationsModule() {
  if (Platform.OS === "web" || isAndroidExpoGo()) {
    return null;
  }

  notificationsModulePromise ??= import("expo-notifications")
    .then((Notifications) => {
      configureNotificationHandler(Notifications);
      return Notifications;
    })
    .catch(() => null);

  return notificationsModulePromise;
}

function configureNotificationHandler(Notifications: ExpoNotificationsModule) {
  if (notificationHandlerConfigured) {
    return;
  }

  Notifications.setNotificationHandler({
    handleNotification: async () => ({
      priority: Notifications.AndroidNotificationPriority.MAX,
      shouldPlaySound: true,
      shouldSetBadge: false,
      shouldShowBanner: true,
      shouldShowList: true,
    }),
  });
  notificationHandlerConfigured = true;
}
