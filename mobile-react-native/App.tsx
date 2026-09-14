import { StatusBar } from "expo-status-bar";
import * as SecureStore from "expo-secure-store";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ActivityIndicator,
  Image,
  type ImageSourcePropType,
  KeyboardAvoidingView,
  Linking,
  Modal,
  Platform,
  Pressable,
  RefreshControl,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  type TextInputProps,
  View,
} from "react-native";

import { API_BASE_URL, authApi, cameraApi, eventApi, getErrorMessage, snapshotUrl } from "./src/api";
import { cameraStatusLabel, cameraStatusTone, eventTitle, formatConfidence, formatDateTime } from "./src/format";
import {
  type NotificationRegistrationState,
  registerForPushNotifications,
  scheduleLocalAlertNotification,
  setupNotificationListeners,
} from "./src/notifications";
import { colors } from "./src/theme";
import type { Camera, DetectionEvent, UserProfile } from "./src/types";

const authTokenKey = "stealth-lens-mobile-token";
const pollIntervalMs = 15_000;

type ActiveTab = "cameras" | "alerts";

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>("cameras");
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [checkingSession, setCheckingSession] = useState(true);
  const [dashboardError, setDashboardError] = useState("");
  const [email, setEmail] = useState("");
  const [events, setEvents] = useState<DetectionEvent[]>([]);
  const [loadingDashboard, setLoadingDashboard] = useState(false);
  const [loginError, setLoginError] = useState("");
  const [notificationState, setNotificationState] = useState<NotificationRegistrationState>({
    message: "Preparando notificações.",
    status: "idle",
  });
  const [password, setPassword] = useState("");
  const [refreshing, setRefreshing] = useState(false);
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);
  const [signingIn, setSigningIn] = useState(false);
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<UserProfile | null>(null);
  const latestEventIdRef = useRef<string | null>(null);

  const loadData = useCallback(
    async (authToken: string, options: { notifyNewAlert: boolean }) => {
      const [nextCameras, nextEvents] = await Promise.all([
        cameraApi.list(authToken),
        eventApi.list(authToken, 80),
      ]);

      setCameras(nextCameras);
      setEvents(nextEvents);

      const latestEvent = nextEvents[0];
      if (!latestEvent) {
        return;
      }

      if (!latestEventIdRef.current) {
        latestEventIdRef.current = latestEvent.id;
        return;
      }

      if (options.notifyNewAlert && latestEvent.id !== latestEventIdRef.current) {
        latestEventIdRef.current = latestEvent.id;
        await scheduleLocalAlertNotification(latestEvent);
      }
    },
    [],
  );

  useEffect(() => {
    let alive = true;

    async function restoreSession() {
      try {
        const storedToken = await SecureStore.getItemAsync(authTokenKey);
        if (!storedToken) {
          return;
        }

        const profile = await authApi.me(storedToken);
        if (!alive) {
          return;
        }

        setToken(storedToken);
        setUser(profile);
        setLoadingDashboard(true);
        await loadData(storedToken, { notifyNewAlert: false });
      } catch (error) {
        await SecureStore.deleteItemAsync(authTokenKey);
        if (alive) {
          setDashboardError(getErrorMessage(error));
        }
      } finally {
        if (alive) {
          setCheckingSession(false);
          setLoadingDashboard(false);
        }
      }
    }

    void restoreSession();
    return () => {
      alive = false;
    };
  }, [loadData]);

  useEffect(() => {
    if (!token) {
      return undefined;
    }

    let stopped = false;
    setNotificationState({
      message: "Registrando aparelho para alertas.",
      status: "idle",
    });

    registerForPushNotifications(token).then((state) => {
      if (!stopped) {
        setNotificationState(state);
      }
    });

    const interval = setInterval(() => {
      loadData(token, { notifyNewAlert: true }).catch((error) => {
        if (!stopped) {
          setDashboardError(getErrorMessage(error));
        }
      });
    }, pollIntervalMs);

    return () => {
      stopped = true;
      clearInterval(interval);
    };
  }, [loadData, token]);

  useEffect(() => {
    if (!token) {
      return undefined;
    }

    let stopped = false;
    let cleanup: () => void = () => undefined;

    setupNotificationListeners({
      onReceived: () => {
        loadData(token, { notifyNewAlert: false }).catch((error) => {
          setDashboardError(getErrorMessage(error));
        });
      },
      onResponse: (eventId) => {
        if (eventId) {
          setActiveTab("alerts");
          setSelectedEventId(eventId);
        }
        loadData(token, { notifyNewAlert: false }).catch((error) => {
          setDashboardError(getErrorMessage(error));
        });
      },
    }).then((nextCleanup) => {
      if (stopped) {
        nextCleanup();
        return;
      }
      cleanup = nextCleanup;
    });

    return () => {
      stopped = true;
      cleanup();
    };
  }, [loadData, token]);

  const selectedEvent = useMemo(
    () => events.find((event) => event.id === selectedEventId) ?? null,
    [events, selectedEventId],
  );

  const onlineCameras = useMemo(
    () => cameras.filter((camera) => String(camera.status).toLowerCase() === "online").length,
    [cameras],
  );

  const handleLogin = useCallback(async () => {
    setLoginError("");
    const cleanEmail = email.trim();
    if (!cleanEmail || !password) {
      setLoginError("Informe email e senha.");
      return;
    }

    setSigningIn(true);
    try {
      const response = await authApi.login(cleanEmail, password);
      await SecureStore.setItemAsync(authTokenKey, response.access_token);
      latestEventIdRef.current = null;
      setToken(response.access_token);
      setUser(response.user);
      setLoadingDashboard(true);
      await loadData(response.access_token, { notifyNewAlert: false });
    } catch (error) {
      setLoginError(getErrorMessage(error));
    } finally {
      setSigningIn(false);
      setLoadingDashboard(false);
    }
  }, [email, loadData, password]);

  const handleLogout = useCallback(async () => {
    await SecureStore.deleteItemAsync(authTokenKey);
    latestEventIdRef.current = null;
    setActiveTab("cameras");
    setCameras([]);
    setDashboardError("");
    setEvents([]);
    setNotificationState({
      message: "Preparando notificações.",
      status: "idle",
    });
    setPassword("");
    setSelectedEventId(null);
    setToken(null);
    setUser(null);
  }, []);

  const handleRefresh = useCallback(async () => {
    if (!token) {
      return;
    }

    setRefreshing(true);
    setDashboardError("");
    try {
      await loadData(token, { notifyNewAlert: false });
    } catch (error) {
      setDashboardError(getErrorMessage(error));
    } finally {
      setRefreshing(false);
    }
  }, [loadData, token]);

  if (checkingSession) {
    return (
      <SafeAreaView style={styles.safeArea}>
        <StatusBar style="light" />
        <View style={styles.centered}>
          <ActivityIndicator color={colors.accent} size="large" />
          <Text style={styles.centeredText}>Carregando sessão</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (!token || !user) {
    return (
      <SafeAreaView style={styles.safeArea}>
        <StatusBar style="light" />
        <KeyboardAvoidingView
          behavior={Platform.OS === "ios" ? "padding" : undefined}
          style={styles.loginScreen}
        >
          <View style={styles.loginHeader}>
            <Text style={styles.brandTitle}>Stealth Lens</Text>
            <Text style={styles.brandSubtitle}>Monitoramento móvel das câmeras F22</Text>
          </View>

          <View style={styles.loginPanel}>
            <LabeledInput
              autoCapitalize="none"
              autoComplete="email"
              keyboardType="email-address"
              label="Email"
              onChangeText={setEmail}
              placeholder="seu@email.com"
              returnKeyType="next"
              value={email}
            />
            <LabeledInput
              label="Senha"
              onChangeText={setPassword}
              onSubmitEditing={handleLogin}
              placeholder="Sua senha"
              returnKeyType="go"
              secureTextEntry
              value={password}
            />
            {loginError ? <Text style={styles.errorText}>{loginError}</Text> : null}
            <Pressable
              disabled={signingIn}
              onPress={handleLogin}
              style={({ pressed }) => [
                styles.primaryButton,
                pressed && styles.primaryButtonPressed,
                signingIn && styles.disabledButton,
              ]}
            >
              {signingIn ? (
                <ActivityIndicator color={colors.background} />
              ) : (
                <Text style={styles.primaryButtonText}>Entrar</Text>
              )}
            </Pressable>
          </View>

          <Text style={styles.apiHint}>API: {API_BASE_URL}</Text>
        </KeyboardAvoidingView>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="light" />
      <View style={styles.header}>
        <View style={styles.headerIdentity}>
          <Text numberOfLines={1} style={styles.headerTitle}>
            Stealth Lens
          </Text>
          <Text numberOfLines={1} style={styles.headerSubtitle}>
            {user.email}
          </Text>
        </View>
        <Pressable onPress={handleLogout} style={styles.secondaryButton}>
          <Text style={styles.secondaryButtonText}>Sair</Text>
        </Pressable>
      </View>

      <ScrollView
        contentContainerStyle={styles.content}
        refreshControl={
          <RefreshControl
            colors={[colors.accent]}
            onRefresh={handleRefresh}
            refreshing={refreshing}
            tintColor={colors.accent}
          />
        }
      >
        <View style={styles.summaryGrid}>
          <StatCard label="Câmeras" value={cameras.length} />
          <StatCard label="Online" tone="ok" value={onlineCameras} />
          <StatCard label="Alertas" tone="danger" value={events.length} />
        </View>

        <View style={styles.notificationStrip}>
          <View style={[styles.notificationDot, notificationTone(notificationState.status)]} />
          <Text style={styles.notificationText}>{notificationState.message}</Text>
        </View>

        {dashboardError ? <Text style={styles.errorText}>{dashboardError}</Text> : null}

        <View style={styles.segmentedControl}>
          <SegmentButton
            active={activeTab === "cameras"}
            label="Câmeras"
            onPress={() => setActiveTab("cameras")}
          />
          <SegmentButton
            active={activeTab === "alerts"}
            label="Alertas"
            onPress={() => setActiveTab("alerts")}
          />
        </View>

        {loadingDashboard ? (
          <View style={styles.loadingPanel}>
            <ActivityIndicator color={colors.accent} />
            <Text style={styles.loadingText}>Atualizando dados</Text>
          </View>
        ) : activeTab === "cameras" ? (
          <CameraList cameras={cameras} />
        ) : (
          <AlertList
            events={events}
            onSelect={(event) => setSelectedEventId(event.id)}
            token={token}
          />
        )}
      </ScrollView>

      <AlertModal event={selectedEvent} onClose={() => setSelectedEventId(null)} token={token} />
    </SafeAreaView>
  );
}

function LabeledInput({ label, style, ...props }: TextInputProps & { label: string }) {
  return (
    <View style={styles.inputGroup}>
      <Text style={styles.inputLabel}>{label}</Text>
      <TextInput
        placeholderTextColor={colors.muted}
        style={[styles.input, style]}
        textContentType={props.secureTextEntry ? "password" : props.textContentType}
        {...props}
      />
    </View>
  );
}

function StatCard({
  label,
  tone = "default",
  value,
}: {
  label: string;
  tone?: "default" | "danger" | "ok";
  value: number | string;
}) {
  return (
    <View style={[styles.statCard, tone === "danger" && styles.statCardDanger, tone === "ok" && styles.statCardOk]}>
      <Text style={styles.statValue}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
    </View>
  );
}

function SegmentButton({
  active,
  label,
  onPress,
}: {
  active: boolean;
  label: string;
  onPress: () => void;
}) {
  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [
        styles.segmentButton,
        active && styles.segmentButtonActive,
        pressed && styles.segmentButtonPressed,
      ]}
    >
      <Text style={[styles.segmentButtonText, active && styles.segmentButtonTextActive]}>{label}</Text>
    </Pressable>
  );
}

function CameraList({ cameras }: { cameras: Camera[] }) {
  if (cameras.length === 0) {
    return <EmptyState title="Nenhuma câmera cadastrada" />;
  }

  return (
    <View style={styles.list}>
      {cameras.map((camera) => (
        <CameraCard camera={camera} key={camera.id} />
      ))}
    </View>
  );
}

function CameraCard({ camera }: { camera: Camera }) {
  const statusTone = cameraStatusTone(camera.status);
  const canOpen = camera.url.startsWith("http://") || camera.url.startsWith("https://");

  return (
    <View style={styles.cameraCard}>
      <CameraPreview camera={camera} />
      <View style={styles.cardBody}>
        <View style={styles.cardHeader}>
          <View style={styles.cardTitleGroup}>
            <Text numberOfLines={1} style={styles.cardTitle}>
              {camera.name}
            </Text>
            <Text numberOfLines={1} style={styles.cardSubtitle}>
              {camera.location || "Sem local informado"}
            </Text>
          </View>
          <View style={styles.statusPill}>
            <View
              style={[
                styles.statusDot,
                statusTone === "ok" && styles.statusDotOk,
                statusTone === "danger" && styles.statusDotDanger,
              ]}
            />
            <Text style={styles.statusText}>{cameraStatusLabel(camera.status)}</Text>
          </View>
        </View>
        <View style={styles.detailGrid}>
          <Detail label="ID externo" value={camera.external_id || camera.id} />
          <Detail label="Último sinal" value={formatDateTime(camera.last_seen_at)} />
        </View>
        <Pressable
          disabled={!canOpen}
          onPress={() => {
            if (canOpen) {
              void Linking.openURL(camera.url);
            }
          }}
          style={({ pressed }) => [
            styles.streamButton,
            pressed && styles.streamButtonPressed,
            !canOpen && styles.disabledButton,
          ]}
        >
          <Text style={styles.streamButtonText}>{canOpen ? "Abrir câmera" : "Stream indisponível"}</Text>
        </Pressable>
      </View>
    </View>
  );
}

function CameraPreview({ camera }: { camera: Camera }) {
  const [failed, setFailed] = useState(false);
  const canPreview =
    !failed && (camera.url.startsWith("http://") || camera.url.startsWith("https://"));

  if (!canPreview) {
    return (
      <View style={styles.cameraPreviewFallback}>
        <Text style={styles.cameraPreviewFallbackText}>Sem preview</Text>
      </View>
    );
  }

  return (
    <Image
      onError={() => setFailed(true)}
      resizeMode="cover"
      source={{ uri: camera.url }}
      style={styles.cameraPreviewImage}
    />
  );
}

function AlertList({
  events,
  onSelect,
  token,
}: {
  events: DetectionEvent[];
  onSelect: (event: DetectionEvent) => void;
  token: string;
}) {
  if (events.length === 0) {
    return <EmptyState title="Nenhum alerta recebido" />;
  }

  return (
    <View style={styles.list}>
      {events.map((event) => (
        <AlertCard event={event} key={event.id} onPress={() => onSelect(event)} token={token} />
      ))}
    </View>
  );
}

function AlertCard({
  event,
  onPress,
  token,
}: {
  event: DetectionEvent;
  onPress: () => void;
  token: string;
}) {
  const source = snapshotImageSource(event, token);

  return (
    <Pressable onPress={onPress} style={({ pressed }) => [styles.alertCard, pressed && styles.alertCardPressed]}>
      {source ? (
        <Image resizeMode="cover" source={source} style={styles.alertThumb} />
      ) : (
        <View style={styles.alertThumbFallback}>
          <Text style={styles.alertThumbFallbackText}>Imagem</Text>
        </View>
      )}
      <View style={styles.alertBody}>
        <View style={styles.alertTitleRow}>
          <Text numberOfLines={1} style={styles.alertTitle}>
            {eventTitle(event)}
          </Text>
          <Text style={styles.confidenceText}>{formatConfidence(event.confidence)}</Text>
        </View>
        <Text numberOfLines={1} style={styles.alertCamera}>
          {event.camera_name || event.camera_external_id}
        </Text>
        <Text style={styles.alertTime}>{formatDateTime(event.occurred_at)}</Text>
      </View>
    </Pressable>
  );
}

function AlertModal({
  event,
  onClose,
  token,
}: {
  event: DetectionEvent | null;
  onClose: () => void;
  token: string;
}) {
  const source = event ? snapshotImageSource(event, token) : null;

  return (
    <Modal animationType="slide" onRequestClose={onClose} transparent visible={Boolean(event)}>
      <View style={styles.modalOverlay}>
        <View style={styles.modalSheet}>
          <View style={styles.modalHeader}>
            <Text numberOfLines={1} style={styles.modalTitle}>
              {event ? eventTitle(event) : "Alerta"}
            </Text>
            <Pressable onPress={onClose} style={styles.modalCloseButton}>
              <Text style={styles.modalCloseText}>Fechar</Text>
            </Pressable>
          </View>

          {source ? (
            <Image resizeMode="cover" source={source} style={styles.modalImage} />
          ) : (
            <View style={styles.modalImageFallback}>
              <Text style={styles.modalImageFallbackText}>Imagem não disponível</Text>
            </View>
          )}

          {event ? (
            <View style={styles.modalDetails}>
              <Detail label="Câmera" value={event.camera_name || event.camera_external_id} />
              <Detail label="Horário" value={formatDateTime(event.occurred_at)} />
              <Detail label="Confiança" value={formatConfidence(event.confidence)} />
              <Detail label="Modelo" value={event.model_alias || "--"} />
            </View>
          ) : null}
        </View>
      </View>
    </Modal>
  );
}

function Detail({ label, value }: { label: string; value?: string }) {
  return (
    <View style={styles.detailItem}>
      <Text style={styles.detailLabel}>{label}</Text>
      <Text numberOfLines={1} style={styles.detailValue}>
        {value || "--"}
      </Text>
    </View>
  );
}

function EmptyState({ title }: { title: string }) {
  return (
    <View style={styles.emptyState}>
      <Text style={styles.emptyTitle}>{title}</Text>
    </View>
  );
}

function snapshotImageSource(event: DetectionEvent, token: string): ImageSourcePropType | null {
  const uri = snapshotUrl(event.snapshot_url);
  if (!uri) {
    return null;
  }
  return {
    headers: {
      Authorization: `Bearer ${token}`,
    },
    uri,
  };
}

function notificationTone(status: NotificationRegistrationState["status"]) {
  if (status === "registered") {
    return styles.statusDotOk;
  }
  if (status === "denied" || status === "failed") {
    return styles.statusDotDanger;
  }
  return styles.statusDotMuted;
}

const styles = StyleSheet.create({
  alertBody: {
    flex: 1,
    gap: 6,
    minWidth: 0,
  },
  alertCamera: {
    color: colors.textSoft,
    fontSize: 14,
  },
  alertCard: {
    alignItems: "center",
    backgroundColor: colors.card,
    borderColor: colors.border,
    borderRadius: 8,
    borderWidth: 1,
    flexDirection: "row",
    gap: 12,
    minHeight: 106,
    padding: 12,
  },
  alertCardPressed: {
    borderColor: colors.accent,
  },
  alertThumb: {
    backgroundColor: colors.mutedPanel,
    borderRadius: 8,
    height: 82,
    width: 82,
  },
  alertThumbFallback: {
    alignItems: "center",
    backgroundColor: colors.mutedPanel,
    borderColor: colors.border,
    borderRadius: 8,
    borderWidth: 1,
    height: 82,
    justifyContent: "center",
    width: 82,
  },
  alertThumbFallbackText: {
    color: colors.muted,
    fontSize: 12,
  },
  alertTime: {
    color: colors.muted,
    fontSize: 13,
  },
  alertTitle: {
    color: colors.text,
    flex: 1,
    fontSize: 17,
    fontWeight: "700",
  },
  alertTitleRow: {
    alignItems: "center",
    flexDirection: "row",
    gap: 10,
    minWidth: 0,
  },
  apiHint: {
    color: colors.muted,
    fontSize: 12,
    lineHeight: 18,
    paddingHorizontal: 28,
    textAlign: "center",
  },
  brandSubtitle: {
    color: colors.textSoft,
    fontSize: 15,
    lineHeight: 22,
    textAlign: "center",
  },
  brandTitle: {
    color: colors.text,
    fontSize: 34,
    fontWeight: "800",
    letterSpacing: 0,
    textAlign: "center",
  },
  cameraCard: {
    backgroundColor: colors.card,
    borderColor: colors.border,
    borderRadius: 8,
    borderWidth: 1,
    overflow: "hidden",
  },
  cameraPreviewFallback: {
    alignItems: "center",
    aspectRatio: 16 / 9,
    backgroundColor: colors.mutedPanel,
    justifyContent: "center",
    width: "100%",
  },
  cameraPreviewFallbackText: {
    color: colors.muted,
    fontSize: 13,
  },
  cameraPreviewImage: {
    aspectRatio: 16 / 9,
    backgroundColor: colors.mutedPanel,
    width: "100%",
  },
  cardBody: {
    gap: 14,
    padding: 14,
  },
  cardHeader: {
    alignItems: "flex-start",
    flexDirection: "row",
    gap: 12,
    justifyContent: "space-between",
  },
  cardSubtitle: {
    color: colors.textSoft,
    fontSize: 14,
    lineHeight: 20,
  },
  cardTitle: {
    color: colors.text,
    fontSize: 18,
    fontWeight: "700",
    lineHeight: 24,
  },
  cardTitleGroup: {
    flex: 1,
    minWidth: 0,
  },
  centered: {
    alignItems: "center",
    flex: 1,
    justifyContent: "center",
    padding: 24,
  },
  centeredText: {
    color: colors.textSoft,
    fontSize: 15,
    marginTop: 14,
  },
  confidenceText: {
    color: colors.accent,
    fontSize: 14,
    fontWeight: "700",
  },
  content: {
    gap: 16,
    padding: 16,
    paddingBottom: 32,
  },
  detailGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
  },
  detailItem: {
    flex: 1,
    minWidth: 128,
  },
  detailLabel: {
    color: colors.muted,
    fontSize: 12,
    marginBottom: 3,
  },
  detailValue: {
    color: colors.textSoft,
    fontSize: 14,
  },
  disabledButton: {
    opacity: 0.55,
  },
  emptyState: {
    alignItems: "center",
    backgroundColor: colors.panel,
    borderColor: colors.border,
    borderRadius: 8,
    borderWidth: 1,
    justifyContent: "center",
    minHeight: 150,
    padding: 22,
  },
  emptyTitle: {
    color: colors.textSoft,
    fontSize: 15,
    textAlign: "center",
  },
  errorText: {
    color: colors.danger,
    fontSize: 13,
    lineHeight: 19,
  },
  header: {
    alignItems: "center",
    backgroundColor: colors.panel,
    borderBottomColor: colors.border,
    borderBottomWidth: 1,
    flexDirection: "row",
    gap: 12,
    justifyContent: "space-between",
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  headerIdentity: {
    flex: 1,
    minWidth: 0,
  },
  headerSubtitle: {
    color: colors.muted,
    fontSize: 13,
    marginTop: 2,
  },
  headerTitle: {
    color: colors.text,
    fontSize: 20,
    fontWeight: "800",
  },
  input: {
    backgroundColor: colors.mutedPanel,
    borderColor: colors.border,
    borderRadius: 8,
    borderWidth: 1,
    color: colors.text,
    fontSize: 16,
    minHeight: 50,
    paddingHorizontal: 14,
  },
  inputGroup: {
    gap: 8,
  },
  inputLabel: {
    color: colors.textSoft,
    fontSize: 13,
    fontWeight: "700",
  },
  list: {
    gap: 12,
  },
  loadingPanel: {
    alignItems: "center",
    backgroundColor: colors.panel,
    borderColor: colors.border,
    borderRadius: 8,
    borderWidth: 1,
    gap: 10,
    justifyContent: "center",
    minHeight: 120,
    padding: 20,
  },
  loadingText: {
    color: colors.textSoft,
    fontSize: 14,
  },
  loginHeader: {
    gap: 8,
    paddingHorizontal: 20,
  },
  loginPanel: {
    backgroundColor: colors.card,
    borderColor: colors.border,
    borderRadius: 8,
    borderWidth: 1,
    gap: 16,
    padding: 18,
    width: "100%",
  },
  loginScreen: {
    flex: 1,
    gap: 22,
    justifyContent: "center",
    padding: 18,
  },
  modalCloseButton: {
    backgroundColor: colors.mutedPanel,
    borderColor: colors.border,
    borderRadius: 8,
    borderWidth: 1,
    paddingHorizontal: 12,
    paddingVertical: 9,
  },
  modalCloseText: {
    color: colors.text,
    fontSize: 13,
    fontWeight: "700",
  },
  modalDetails: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 14,
  },
  modalHeader: {
    alignItems: "center",
    flexDirection: "row",
    gap: 12,
    justifyContent: "space-between",
  },
  modalImage: {
    aspectRatio: 16 / 10,
    backgroundColor: colors.mutedPanel,
    borderRadius: 8,
    width: "100%",
  },
  modalImageFallback: {
    alignItems: "center",
    aspectRatio: 16 / 10,
    backgroundColor: colors.mutedPanel,
    borderColor: colors.border,
    borderRadius: 8,
    borderWidth: 1,
    justifyContent: "center",
    width: "100%",
  },
  modalImageFallbackText: {
    color: colors.muted,
    fontSize: 14,
  },
  modalOverlay: {
    backgroundColor: "rgba(0, 0, 0, 0.68)",
    flex: 1,
    justifyContent: "flex-end",
  },
  modalSheet: {
    backgroundColor: colors.card,
    borderColor: colors.border,
    borderTopLeftRadius: 8,
    borderTopRightRadius: 8,
    borderWidth: 1,
    gap: 16,
    maxHeight: "88%",
    padding: 16,
  },
  modalTitle: {
    color: colors.text,
    flex: 1,
    fontSize: 20,
    fontWeight: "800",
  },
  notificationDot: {
    backgroundColor: colors.muted,
    borderRadius: 5,
    height: 10,
    width: 10,
  },
  notificationStrip: {
    alignItems: "center",
    backgroundColor: colors.panel,
    borderColor: colors.border,
    borderRadius: 8,
    borderWidth: 1,
    flexDirection: "row",
    gap: 10,
    paddingHorizontal: 12,
    paddingVertical: 11,
  },
  notificationText: {
    color: colors.textSoft,
    flex: 1,
    fontSize: 13,
    lineHeight: 18,
  },
  primaryButton: {
    alignItems: "center",
    backgroundColor: colors.accent,
    borderRadius: 8,
    justifyContent: "center",
    minHeight: 50,
    paddingHorizontal: 16,
  },
  primaryButtonPressed: {
    opacity: 0.85,
  },
  primaryButtonText: {
    color: colors.background,
    fontSize: 16,
    fontWeight: "800",
  },
  safeArea: {
    backgroundColor: colors.background,
    flex: 1,
  },
  secondaryButton: {
    backgroundColor: colors.mutedPanel,
    borderColor: colors.border,
    borderRadius: 8,
    borderWidth: 1,
    paddingHorizontal: 14,
    paddingVertical: 10,
  },
  secondaryButtonText: {
    color: colors.text,
    fontSize: 13,
    fontWeight: "700",
  },
  segmentButton: {
    alignItems: "center",
    borderRadius: 7,
    flex: 1,
    justifyContent: "center",
    minHeight: 42,
    paddingHorizontal: 10,
  },
  segmentButtonActive: {
    backgroundColor: colors.accent,
  },
  segmentButtonPressed: {
    opacity: 0.86,
  },
  segmentButtonText: {
    color: colors.textSoft,
    fontSize: 14,
    fontWeight: "700",
  },
  segmentButtonTextActive: {
    color: colors.background,
  },
  segmentedControl: {
    backgroundColor: colors.panel,
    borderColor: colors.border,
    borderRadius: 8,
    borderWidth: 1,
    flexDirection: "row",
    padding: 4,
  },
  statCard: {
    backgroundColor: colors.card,
    borderColor: colors.border,
    borderRadius: 8,
    borderWidth: 1,
    flex: 1,
    minHeight: 82,
    minWidth: 96,
    padding: 12,
  },
  statCardDanger: {
    borderColor: "rgba(251, 93, 93, 0.55)",
  },
  statCardOk: {
    borderColor: "rgba(74, 222, 128, 0.55)",
  },
  statLabel: {
    color: colors.muted,
    fontSize: 12,
    marginTop: 5,
  },
  statValue: {
    color: colors.text,
    fontSize: 28,
    fontWeight: "800",
  },
  statusDot: {
    backgroundColor: colors.muted,
    borderRadius: 5,
    height: 10,
    width: 10,
  },
  statusDotDanger: {
    backgroundColor: colors.danger,
  },
  statusDotMuted: {
    backgroundColor: colors.amber,
  },
  statusDotOk: {
    backgroundColor: colors.success,
  },
  statusPill: {
    alignItems: "center",
    backgroundColor: colors.mutedPanel,
    borderColor: colors.border,
    borderRadius: 8,
    borderWidth: 1,
    flexDirection: "row",
    gap: 7,
    paddingHorizontal: 9,
    paddingVertical: 7,
  },
  statusText: {
    color: colors.textSoft,
    fontSize: 12,
    fontWeight: "700",
  },
  streamButton: {
    alignItems: "center",
    backgroundColor: colors.mutedPanel,
    borderColor: colors.border,
    borderRadius: 8,
    borderWidth: 1,
    justifyContent: "center",
    minHeight: 42,
    paddingHorizontal: 12,
  },
  streamButtonPressed: {
    borderColor: colors.accent,
  },
  streamButtonText: {
    color: colors.text,
    fontSize: 14,
    fontWeight: "700",
  },
  summaryGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
  },
});
