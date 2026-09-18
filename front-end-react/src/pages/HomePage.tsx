import { useEffect, useMemo, useState } from "react";
import type { ChangeEvent, FormEvent } from "react";
import {
  AlertTriangle,
  Ban,
  Camera as CameraIcon,
  CheckCircle2,
  Clock3,
  Copy,
  Eye,
  Gauge,
  ImageOff,
  KeyRound,
  MapPin,
  Pencil,
  Plus,
  RefreshCw,
  RotateCcw,
  Save,
  Trash2,
  Video,
  X,
} from "lucide-react";
import { Link } from "react-router-dom";
import ActionButton from "../components/ui/ActionButton";
import FormField from "../components/ui/FormField";
import PageHeader from "../components/ui/PageHeader";
import Panel from "../components/ui/Panel";
import StatusBadge from "../components/ui/StatusBadge";
import { API_BASE_URL, DEFAULT_API_BASE_URL } from "../config/api";
import {
  agentKeyApi,
  authApi,
  cameraApi,
  detectionEventApi,
  fetchApiAssetBlob,
  getErrorMessage,
} from "../lib/apiClient";
import type { AgentAccessKey, DetectionEvent, UserProfile } from "../lib/apiClient";
import type { Camera, CameraPayload, CameraStatus } from "../types/camera";

const agentKeyStorageKey = "stealth-lens-agent-key";

const initialForm: CameraPayload = {
  location: "",
  name: "",
  status: "unknown",
  url: "",
};

const validCameraStatuses: CameraStatus[] = ["unknown", "online", "offline"];
const allowedCameraProtocols = ["rtsp:", "rtsps:", "http:", "https:", "local:"];

function getStatusTone(
  status: string,
): "danger" | "neutral" | "success" | "warning" {
  if (status === "online") return "success";
  if (status === "offline") return "danger";
  if (status === "unknown") return "warning";
  return "neutral";
}

function formatStatus(status: string) {
  const labels: Record<string, string> = {
    offline: "Offline",
    online: "Online",
    unknown: "Sem sinal",
  };

  return labels[status] ?? status;
}

function isValidIPv4(value: string) {
  const parts = value.split(".");
  if (parts.length !== 4) return false;

  return parts.every((part) => {
    if (!/^\d{1,3}$/.test(part)) return false;
    const number = Number(part);
    return number >= 0 && number <= 255;
  });
}

function isValidCameraSource(value: string) {
  const source = value.trim();
  if (!source) return false;
  if (source === "0" || source.toLowerCase().startsWith("local://")) return true;
  if (isValidIPv4(source)) return true;

  try {
    const parsed = new URL(source);
    return Boolean(parsed.hostname) && allowedCameraProtocols.includes(parsed.protocol);
  } catch {
    return false;
  }
}

function validateCameraForm(payload: CameraPayload) {
  const errors: string[] = [];

  if (!payload.name.trim()) {
    errors.push("Informe o nome da câmera.");
  }

  if (!payload.location.trim()) {
    errors.push("Informe o local da câmera.");
  }

  if (!validCameraStatuses.includes(payload.status)) {
    errors.push("Selecione um status válido para a câmera.");
  }

  return errors;
}

function normalizeCameraPayload(payload: CameraPayload): CameraPayload {
  const trimmedUrl = payload.url.trim();

  return {
    location: payload.location.trim(),
    name: payload.name.trim(),
    status: payload.status,
    url: isValidCameraSource(trimmedUrl) ? trimmedUrl : DEFAULT_API_BASE_URL,
  };
}

function formatCameraSubmitError(error: unknown) {
  const message = getErrorMessage(error).toLowerCase();

  if (
    message.includes("dados da camera invalidos") ||
    message.includes("dados da câmera inválidos") ||
    message.includes("corpo invalido") ||
    message.includes("corpo inválido")
  ) {
    return "Não foi possível salvar a câmera. Revise nome, local e status antes de tentar novamente.";
  }

  if (message.includes("missing authentication token") || message.includes("401")) {
    return "Sua sessão expirou ou não foi encontrada. Faça login novamente para salvar câmeras.";
  }

  return getErrorMessage(error);
}

function formatEventType(event: DetectionEvent) {
  const raw = event.label || event.event_type || "alerta";
  const normalized = raw.trim().toLowerCase();
  const labels: Record<string, string> = {
    cigarette: "Cigarro",
    fire: "Fogo",
    knife: "Faca",
    person: "Pessoa",
    smoke: "Fumaça",
    smoking: "Pessoa fumando",
  };

  if (labels[normalized]) {
    return labels[normalized];
  }

  const clean = raw.replace(/[_-]+/g, " ").trim();
  return clean ? clean.charAt(0).toUpperCase() + clean.slice(1) : "Alerta";
}

function formatEventDate(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Horário indisponível";
  }
  return date.toLocaleString("pt-BR", {
    dateStyle: "short",
    timeStyle: "medium",
  });
}

function formatConfidence(value: number) {
  if (!Number.isFinite(value)) {
    return "--";
  }
  return `${Math.round(value * 100)}%`;
}

function getEventTone(eventType: string): "danger" | "neutral" | "success" | "warning" {
  const normalized = eventType.toLowerCase();
  if (normalized.includes("smoking") || normalized.includes("knife") || normalized.includes("fire")) {
    return "danger";
  }
  if (normalized.includes("smoke")) {
    return "warning";
  }
  return "neutral";
}

function HomePage() {
  const [agentKeyName, setAgentKeyName] = useState("Distribuído Stealth Lens");
  const [agentKeys, setAgentKeys] = useState<AgentAccessKey[]>([]);
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [events, setEvents] = useState<DetectionEvent[]>([]);
  const [eventsError, setEventsError] = useState("");
  const [form, setForm] = useState<CameraPayload>(initialForm);
  const [keyError, setKeyError] = useState("");
  const [keySuccess, setKeySuccess] = useState("");
  const [latestAgentKey, setLatestAgentKey] = useState(
    () => localStorage.getItem(agentKeyStorageKey) ?? "",
  );
  const [loading, setLoading] = useState(false);
  const [loadingEvents, setLoadingEvents] = useState(false);
  const [loadingKeys, setLoadingKeys] = useState(false);
  const [savingKey, setSavingKey] = useState(false);
  const [saving, setSaving] = useState(false);
  const [selectedEvent, setSelectedEvent] = useState<DetectionEvent | null>(null);
  const [success, setSuccess] = useState("");
  const [user, setUser] = useState<UserProfile | null>(null);

  const totals = useMemo(
    () => ({
      all: cameras.length,
      alerts: events.length,
      offline: cameras.filter((camera) => camera.status === "offline").length,
      online: cameras.filter((camera) => camera.status === "online").length,
    }),
    [cameras, events.length],
  );

  useEffect(() => {
    void restoreSession();
    void loadCameras();
  }, []);

  useEffect(() => {
    localStorage.setItem(agentKeyStorageKey, latestAgentKey);
  }, [latestAgentKey]);

  async function restoreSession() {
    try {
      const currentUser = await authApi.me();
      setUser(currentUser);
      await Promise.all([loadAgentKeys(true), loadEvents(true)]);
    } catch {
      setUser(null);
    }
  }

  async function loadAgentKeys(quiet = false) {
    setLoadingKeys(true);
    if (!quiet) {
      setKeyError("");
      setKeySuccess("");
    }

    try {
      const result = await agentKeyApi.list();
      setAgentKeys(result);
    } catch (loadError) {
      if (!quiet) {
        setKeyError(getErrorMessage(loadError));
      }
    } finally {
      setLoadingKeys(false);
    }
  }

  async function loadCameras() {
    setLoading(true);
    setError("");

    try {
      const result = await cameraApi.list();
      setCameras(result);
    } catch (loadError) {
      setError(getErrorMessage(loadError));
    } finally {
      setLoading(false);
    }
  }

  async function loadEvents(quiet = false) {
    setLoadingEvents(true);
    if (!quiet) {
      setEventsError("");
    }

    try {
      const result = await detectionEventApi.list(80);
      setEvents(result);
    } catch (loadError) {
      if (!quiet) {
        setEventsError(getErrorMessage(loadError));
      }
    } finally {
      setLoadingEvents(false);
    }
  }

  async function refreshDashboard() {
    await Promise.all([
      loadCameras(),
      user ? loadEvents() : Promise.resolve(),
    ]);
  }

  function updateField(field: keyof CameraPayload, value: string) {
    setForm((current) => ({
      ...current,
      [field]: field === "status" ? (value as CameraStatus) : value,
    }));
  }

  function resetForm() {
    setEditingId(null);
    setForm(initialForm);
  }

  function selectCamera(camera: Camera) {
    setEditingId(camera.id);
    setForm({
      location: camera.location,
      name: camera.name,
      status: (camera.status as CameraStatus) || "unknown",
      url: camera.url,
    });
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSaving(true);
    setError("");
    setSuccess("");

    const validationErrors = validateCameraForm(form);
    if (validationErrors.length > 0) {
      setError(validationErrors.join(" "));
      setSaving(false);
      return;
    }

    try {
      const payload = normalizeCameraPayload(form);
      const usedDefaultUrl = payload.url === DEFAULT_API_BASE_URL && !isValidCameraSource(form.url);

      if (editingId) {
        await cameraApi.update(editingId, payload);
        setSuccess(
          usedDefaultUrl
            ? `Câmera atualizada com sucesso. A URL inválida foi substituída por ${DEFAULT_API_BASE_URL}.`
            : "Câmera atualizada com sucesso.",
        );
      } else {
        await cameraApi.create(payload);
        setSuccess(
          usedDefaultUrl
            ? `Câmera cadastrada com sucesso. A URL inválida foi substituída por ${DEFAULT_API_BASE_URL}.`
            : "Câmera cadastrada com sucesso.",
        );
      }

      resetForm();
      await loadCameras();
    } catch (submitError) {
      setError(formatCameraSubmitError(submitError));
    } finally {
      setSaving(false);
    }
  }

  async function deleteCamera(camera: Camera) {
    const shouldDelete = window.confirm(`Excluir a câmera "${camera.name}"?`);
    if (!shouldDelete) return;

    setError("");
    setSuccess("");

    try {
      await cameraApi.delete(camera.id);
      setSuccess("Câmera excluída.");
      if (editingId === camera.id) resetForm();
      await loadCameras();
    } catch (deleteError) {
      setError(getErrorMessage(deleteError));
    }
  }

  async function createAgentKey(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSavingKey(true);
    setKeyError("");
    setKeySuccess("");

    try {
      const createdKey = await agentKeyApi.create(agentKeyName);
      setLatestAgentKey(createdKey.access_key ?? "");
      setKeySuccess("Chave criada. Guarde a chave completa antes de sair da tela.");
      await loadAgentKeys(true);
    } catch (createError) {
      setKeyError(getErrorMessage(createError));
    } finally {
      setSavingKey(false);
    }
  }

  async function copyLatestAgentKey() {
    if (!latestAgentKey) return;

    try {
      await navigator.clipboard.writeText(latestAgentKey);
      setKeyError("");
      setKeySuccess("Chave copiada.");
    } catch {
      setKeyError("Não foi possível copiar automaticamente.");
    }
  }

  async function revokeAgentKey(key: AgentAccessKey) {
    const shouldRevoke = window.confirm(`Revogar a chave "${key.name}"?`);
    if (!shouldRevoke) return;

    setKeyError("");
    setKeySuccess("");

    try {
      await agentKeyApi.revoke(key.id);
      setKeySuccess("Chave revogada.");
      await loadAgentKeys(true);
    } catch (revokeError) {
      setKeyError(getErrorMessage(revokeError));
    }
  }

  return (
    <>
      <PageHeader

        title="Painel de câmeras"
        description="Gerencie câmeras, pontos de monitoramento e estados operacionais em uma tela direta."
        action={
          <div className="flex flex-col gap-3 sm:flex-row">
            <ActionButton
              disabled={loading || loadingEvents}
              icon={RefreshCw}
              onClick={() => void refreshDashboard()}
              variant="secondary"
            >
              Atualizar
            </ActionButton>
            <Link
              className="inline-flex min-h-11 items-center justify-center rounded-lg bg-[#202140] px-4 py-2.5 text-sm font-semibold text-white shadow-[0_12px_26px_rgba(32,33,64,0.32)] transition-all duration-200 ease-out hover:bg-[#262750] dark:bg-gradient-to-r dark:from-cyan-300 dark:via-cyan-400 dark:to-blue-500 dark:text-slate-950 dark:shadow-[0_16px_34px_rgba(14,165,233,0.24)] dark:hover:brightness-110"
              to="/login"
            >
              {user ? user.name : "Entrar"}
            </Link>
          </div>
        }
      />

      <div className="mx-auto grid max-w-7xl gap-6 px-4 pb-12 sm:px-6">
        <section className="grid gap-4 md:grid-cols-4">
          {[
            {
              label: "Total",
              value: totals.all,
              detail: "câmeras cadastradas",
              accent: "from-cyan-500 to-blue-500",
              chip: "bg-cyan-700/10 text-cyan-700 dark:bg-cyan-300/12 dark:text-cyan-100",
            },
            {
              label: "Online",
              value: totals.online,
              detail: "pontos ativos",
              accent: "from-emerald-500 to-teal-500",
              chip: "bg-emerald-700/10 text-emerald-700 dark:bg-emerald-300/12 dark:text-emerald-100",
            },
            {
              label: "Offline",
              value: totals.offline,
              detail: "precisam de atenção",
              accent: "from-rose-500 to-red-500",
              chip: "bg-red-700/10 text-red-700 dark:bg-red-300/12 dark:text-red-100",
            },
            { label: "Total", value: totals.all, detail: "câmeras cadastradas" },
            { label: "Online", value: totals.online, detail: "pontos ativos" },
            { label: "Offline", value: totals.offline, detail: "precisam de atenção" },
            { label: "Alertas", value: totals.alerts, detail: "eventos recentes" },
          ].map((item) => (
            <article
              className="surface relative overflow-hidden rounded-xl p-5 transition hover:-translate-y-0.5 hover:shadow-[0_24px_60px_rgba(2,44,80,0.16)] dark:hover:shadow-[0_24px_60px_rgba(0,0,0,0.45)]"
              key={item.label}
            >
              <div
                className={`absolute inset-x-0 top-0 h-1 bg-gradient-to-r ${item.accent}`}
              />
              <p className="text-sm font-semibold text-slate-500 dark:text-slate-400">{item.label}</p>
              <strong className="mt-2 block font-display text-4xl text-slate-900 dark:text-white">
                {item.value}
              </strong>
              <p
                className={`mt-2 inline-flex rounded-full px-2.5 py-0.5 text-xs font-semibold ${item.chip}`}
              >
                {item.detail}
              </p>
            </article>
          ))}
        </section>

        {(error || success) && (
          <div
            className={[
              "rounded-lg border px-4 py-3 text-sm",
              error
                ? "border-red-500/30 bg-red-400/12 text-red-700 dark:border-red-300/20 dark:bg-red-300/10 dark:text-red-100"
                : "border-emerald-500/30 bg-emerald-400/12 text-emerald-700 dark:border-emerald-300/20 dark:bg-emerald-300/10 dark:text-emerald-100",
            ].join(" ")}
          >
            {error || success}
          </div>
        )}

        <Panel
          title="Alertas recentes"
          description="Eventos recebidos pela API, com horário, câmera, confiança e snapshot quando o agente enviou a imagem."
        >
          {eventsError ? (
            <div className="mb-4 rounded-lg border border-red-300/20 bg-red-300/10 px-4 py-3 text-sm text-red-100">
              {eventsError}
            </div>
          ) : null}

          {loadingEvents ? (
            <div className="rounded-lg border border-white/[0.08] bg-white/[0.03] px-4 py-8 text-sm text-slate-400">
              Carregando alertas...
            </div>
          ) : events.length === 0 ? (
            <div className="rounded-lg border border-white/[0.08] bg-white/[0.03] px-4 py-8 text-sm text-slate-400">
              Nenhum alerta recebido ainda.
            </div>
          ) : (
            <div className="grid gap-4 lg:grid-cols-2">
              {events.slice(0, 12).map((event) => (
                <AlertCard
                  event={event}
                  key={event.id}
                  onOpen={() => setSelectedEvent(event)}
                />
              ))}
            </div>
          )}
        </Panel>

        <section className="grid gap-6 xl:grid-cols-[0.9fr_1.1fr]">
          <Panel
            title="Chave do distribuído"
            description="Crie a chave usada pelo app local para sincronizar câmeras e eventos."
          >
            {(keyError || keySuccess) && (
              <div
                className={[
                  "mb-4 rounded-lg border px-4 py-3 text-sm",
                  keyError
                    ? "border-red-500/30 bg-red-400/12 text-red-700 dark:border-red-300/20 dark:bg-red-300/10 dark:text-red-100"
                    : "border-emerald-500/30 bg-emerald-400/12 text-emerald-700 dark:border-emerald-300/20 dark:bg-emerald-300/10 dark:text-emerald-100",
                ].join(" ")}
              >
                {keyError || keySuccess}
              </div>
            )}

            <form className="grid gap-4" onSubmit={createAgentKey}>
              <FormField
                icon={KeyRound}
                id="agent-key-name"
                label="Nome da chave"
                name="agentKeyName"
                onChange={(event: ChangeEvent<HTMLInputElement>) =>
                  setAgentKeyName(event.target.value)
                }
                placeholder="Distribuído da recepção"
                required
                value={agentKeyName}
              />

              {latestAgentKey ? (
                <div className="rounded-lg border border-cyan-500/25 bg-cyan-400/12 p-4 dark:border-cyan-300/20 dark:bg-cyan-300/8">
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                    <div className="min-w-0">
                      <p className="text-sm font-semibold text-cyan-800 dark:text-cyan-100">
                        Chave gerada
                      </p>
                      <p className="mt-2 break-all font-mono text-xs leading-6 text-slate-600 dark:text-slate-200">
                        {latestAgentKey}
                      </p>
                    </div>
                    <ActionButton
                      className="shrink-0"
                      icon={Copy}
                      onClick={() => void copyLatestAgentKey()}
                      variant="secondary"
                    >
                      Copiar
                    </ActionButton>
                  </div>
                </div>
              ) : null}

              <div className="flex flex-col gap-3 sm:flex-row">
                <ActionButton disabled={savingKey} icon={KeyRound} type="submit">
                  {savingKey ? "Gerando..." : "Gerar chave"}
                </ActionButton>
                <ActionButton
                  disabled={loadingKeys}
                  icon={RefreshCw}
                  onClick={() => void loadAgentKeys()}
                  variant="ghost"
                >
                  Listar chaves
                </ActionButton>
              </div>
            </form>
          </Panel>

          <Panel
            title="Chaves cadastradas"
            description="A chave completa aparece somente quando é criada; depois, a lista mostra apenas o prefixo."
          >
            <div className="overflow-hidden rounded-lg border border-slate-900/10 dark:border-white/[0.08]">
              <div className="grid grid-cols-[1fr_auto] gap-4 border-b border-slate-900/10 bg-slate-900/[0.04] px-4 py-3 text-sm font-semibold text-slate-600 dark:border-white/[0.08] dark:bg-white/[0.04] dark:text-slate-300">
                <span>Chave</span>
                <span>Ações</span>
              </div>

              {loadingKeys ? (
                <p className="px-4 py-6 text-sm text-slate-500 dark:text-slate-400">Carregando chaves...</p>
              ) : agentKeys.length === 0 ? (
                <p className="px-4 py-6 text-sm text-slate-500 dark:text-slate-400">
                  Nenhuma chave cadastrada ainda.
                </p>
              ) : (
                <div className="divide-y divide-slate-900/10 dark:divide-white/[0.08]">
                  {agentKeys.map((key) => (
                    <div
                      className="grid gap-4 px-4 py-4 md:grid-cols-[minmax(0,1fr)_auto] md:items-center"
                      key={key.id}
                    >
                      <div className="min-w-0">
                        <div className="flex flex-wrap items-center gap-2">
                          <h3 className="font-semibold text-slate-900 dark:text-white">{key.name}</h3>
                          <StatusBadge tone={key.revoked_at ? "danger" : "success"}>
                            {key.revoked_at ? "Revogada" : "Ativa"}
                          </StatusBadge>
                        </div>
                        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
                          Prefixo: {key.key_prefix}
                        </p>
                        {key.last_used_at ? (
                          <p className="mt-1 text-xs text-slate-400 dark:text-slate-500">
                            Último uso: {new Date(key.last_used_at).toLocaleString()}
                          </p>
                        ) : null}
                      </div>

                      <div className="flex gap-2 md:justify-end">
                        {!key.revoked_at ? (
                          <button
                            className="inline-flex size-10 items-center justify-center rounded-lg border border-red-500/30 bg-red-50 text-red-600 shadow-sm transition-colors duration-200 hover:border-red-500/45 hover:bg-red-100 dark:border-red-300/20 dark:bg-red-300/10 dark:text-red-100 dark:shadow-none dark:hover:border-red-300/36 dark:hover:bg-red-300/16 dark:hover:text-red-100"
                            type="button"
                            aria-label={`Revogar ${key.name}`}
                            title="Revogar"
                            onClick={() => void revokeAgentKey(key)}
                          >
                            <Ban className="size-4" />
                          </button>
                        ) : null}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </Panel>
        </section>

        <section className="grid gap-6 xl:grid-cols-[0.9fr_1.1fr]">
          <Panel
            title={editingId ? "Editar câmera" : "Nova câmera"}
            description={`API configurada: ${API_BASE_URL}`}
          >
            <form className="grid gap-4" noValidate onSubmit={handleSubmit}>
              <FormField
                icon={Video}
                id="camera-name"
                label="Nome"
                name="name"
                onChange={(event: ChangeEvent<HTMLInputElement>) =>
                  updateField("name", event.target.value)
                }
                placeholder="Entrada principal"
                required
                value={form.name}
              />
              <FormField
                icon={MapPin}
                id="camera-location"
                label="Local"
                name="location"
                onChange={(event: ChangeEvent<HTMLInputElement>) =>
                  updateField("location", event.target.value)
                }
                placeholder="Portão, recepção, estacionamento..."
                required
                value={form.location}
              />
              <FormField
                icon={CameraIcon}
                id="camera-url"
                label="URL/IP"
                name="url"
                onChange={(event: ChangeEvent<HTMLInputElement>) =>
                  updateField("url", event.target.value)
                }
                placeholder="rtsp://192.168.1.10:554/stream"
                required
                value={form.url}
              />
              <label className="grid gap-2 text-sm font-medium text-slate-700 dark:text-slate-200" htmlFor="camera-status">
                Status
                <select
                  className="input-shell min-h-11 w-full rounded-lg px-3 py-2.5 text-sm"
                  id="camera-status"
                  onChange={(event) => updateField("status", event.target.value)}
                  value={form.status}
                >
                  <option value="unknown">Sem sinal</option>
                  <option value="online">Online</option>
                  <option value="offline">Offline</option>
                </select>
              </label>

              <div className="flex flex-col gap-3 sm:flex-row">
                <ActionButton
                  disabled={saving}
                  icon={editingId ? Save : Plus}
                  type="submit"
                >
                  {saving ? "Salvando..." : editingId ? "Salvar edição" : "Cadastrar"}
                </ActionButton>
                {editingId ? (
                  <ActionButton icon={RotateCcw} onClick={resetForm} variant="ghost">
                    Cancelar
                  </ActionButton>
                ) : null}
              </div>
            </form>
          </Panel>

          <Panel
            title="Câmeras cadastradas"
            description="Registros ativos da operação, com status, localização e origem de sincronização quando houver."
          >
            <div className="overflow-hidden rounded-lg border border-slate-900/10 dark:border-white/[0.08]">
              <div className="grid grid-cols-[1fr_auto] gap-4 border-b border-slate-900/10 bg-slate-900/[0.04] px-4 py-3 text-sm font-semibold text-slate-600 dark:border-white/[0.08] dark:bg-white/[0.04] dark:text-slate-300">
                <span>Câmera</span>
                <span>Ações</span>
              </div>

              {loading ? (
                <p className="px-4 py-6 text-sm text-slate-500 dark:text-slate-400">Carregando câmeras...</p>
              ) : cameras.length === 0 ? (
                <p className="px-4 py-6 text-sm text-slate-500 dark:text-slate-400">
                  Nenhuma câmera carregada ainda.
                </p>
              ) : (
                <div className="divide-y divide-slate-900/10 dark:divide-white/[0.08]">
                  {cameras.map((camera) => (
                    <div
                      className="grid gap-4 px-4 py-4 md:grid-cols-[minmax(0,1fr)_auto] md:items-center"
                      key={camera.id}
                    >
                      <div className="min-w-0">
                        <div className="flex flex-wrap items-center gap-2">
                          <h3 className="font-semibold text-slate-900 dark:text-white">{camera.name}</h3>
                          <StatusBadge tone={getStatusTone(camera.status)}>
                            {formatStatus(camera.status)}
                          </StatusBadge>
                        </div>
                        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">{camera.location}</p>
                        <p className="mt-1 break-all text-xs text-cyan-700/90 dark:text-cyan-100/80">
                          {camera.url}
                        </p>
                        {camera.external_id ? (
                          <p className="mt-1 text-xs text-emerald-700/90 dark:text-emerald-100/80">
                            Sync: {camera.agent_id}/{camera.external_id}
                          </p>
                        ) : null}
                      </div>

                      <div className="flex gap-2 md:justify-end">
                        <button
                          className="inline-flex size-10 items-center justify-center rounded-lg border border-slate-900/15 bg-white text-slate-700 shadow-sm transition-colors duration-200 hover:border-cyan-700/35 hover:bg-cyan-700/[0.06] hover:text-cyan-800 dark:border-white/10 dark:bg-white/[0.03] dark:text-slate-200 dark:shadow-none dark:hover:border-cyan-300/28 dark:hover:bg-white/[0.06] dark:hover:text-cyan-100"
                          type="button"
                          aria-label={`Editar ${camera.name}`}
                          title="Editar"
                          onClick={() => selectCamera(camera)}
                        >
                          <Pencil className="size-4" />
                        </button>
                        <button
                          className="inline-flex size-10 items-center justify-center rounded-lg border border-red-500/30 bg-red-50 text-red-600 shadow-sm transition-colors duration-200 hover:border-red-500/45 hover:bg-red-100 dark:border-red-300/20 dark:bg-red-300/10 dark:text-red-100 dark:shadow-none dark:hover:border-red-300/36 dark:hover:bg-red-300/16 dark:hover:text-red-100"
                          type="button"
                          aria-label={`Excluir ${camera.name}`}
                          title="Excluir"
                          onClick={() => void deleteCamera(camera)}
                        >
                          <Trash2 className="size-4" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </Panel>
        </section>
      </div>
      <AlertModal event={selectedEvent} onClose={() => setSelectedEvent(null)} />
    </>
  );
}

function AlertCard({
  event,
  onOpen,
}: {
  event: DetectionEvent;
  onOpen: () => void;
}) {
  return (
    <article className="grid gap-4 rounded-lg border border-white/[0.08] bg-white/[0.035] p-3 sm:grid-cols-[148px_minmax(0,1fr)]">
      <SnapshotImage
        alt={`Snapshot do alerta ${formatEventType(event)}`}
        className="h-36 w-full rounded-lg object-cover sm:h-full"
        fallbackClassName="h-36 w-full rounded-lg sm:h-full"
        snapshotPath={event.snapshot_url}
        publicSnapshotPath={event.public_snapshot_url}
      />
      <div className="flex min-w-0 flex-col gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <StatusBadge tone={getEventTone(event.event_type)}>
            {formatEventType(event)}
          </StatusBadge>
          {event.snapshot_size ? (
            <StatusBadge tone="success">Com imagem</StatusBadge>
          ) : (
            <StatusBadge tone="warning">Sem imagem</StatusBadge>
          )}
        </div>

        <div className="min-w-0">
          <h3 className="truncate font-display text-xl text-white">
            {event.camera_name || event.camera_external_id}
          </h3>
          <p className="mt-1 truncate text-sm text-slate-400">
            {event.camera_external_id}
          </p>
        </div>

        <div className="grid gap-2 text-sm text-slate-300 sm:grid-cols-2">
          <span className="inline-flex items-center gap-2">
            <Clock3 className="size-4 text-cyan-200" />
            {formatEventDate(event.occurred_at)}
          </span>
          <span className="inline-flex items-center gap-2">
            <Gauge className="size-4 text-cyan-200" />
            {formatConfidence(event.confidence)}
          </span>
        </div>

        <button
          className="mt-auto inline-flex min-h-10 items-center justify-center gap-2 rounded-lg border border-cyan-300/20 bg-cyan-300/10 px-3 py-2 text-sm font-semibold text-cyan-50 transition hover:border-cyan-300/38 hover:bg-cyan-300/16"
          type="button"
          onClick={onOpen}
        >
          <Eye className="size-4" />
          Ver alerta
        </button>
      </div>
    </article>
  );
}

function AlertModal({
  event,
  onClose,
}: {
  event: DetectionEvent | null;
  onClose: () => void;
}) {
  if (!event) {
    return null;
  }

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-slate-950/78 px-4 py-6 backdrop-blur-sm">
      <div className="surface max-h-[92vh] w-full max-w-5xl overflow-hidden rounded-lg">
        <div className="flex items-start justify-between gap-4 border-b border-white/[0.08] px-5 py-4">
          <div className="min-w-0">
            <StatusBadge tone={getEventTone(event.event_type)}>
              {formatEventType(event)}
            </StatusBadge>
            <h2 className="mt-3 truncate font-display text-2xl text-white">
              {event.camera_name || event.camera_external_id}
            </h2>
          </div>
          <button
            className="inline-flex size-10 shrink-0 items-center justify-center rounded-lg border border-white/10 bg-white/[0.04] text-slate-100 transition hover:border-cyan-300/28"
            type="button"
            aria-label="Fechar alerta"
            title="Fechar"
            onClick={onClose}
          >
            <X className="size-5" />
          </button>
        </div>

        <div className="grid max-h-[calc(92vh-88px)] gap-5 overflow-y-auto p-5 lg:grid-cols-[minmax(0,1.25fr)_0.75fr]">
          <SnapshotImage
            alt={`Imagem do alerta em ${event.camera_name}`}
            className="aspect-video w-full rounded-lg object-cover"
            fallbackClassName="aspect-video w-full rounded-lg"
            snapshotPath={event.snapshot_url}
            publicSnapshotPath={event.public_snapshot_url}
          />

          <div className="grid content-start gap-4">
            <AlertDetail icon={Clock3} label="Horário" value={formatEventDate(event.occurred_at)} />
            <AlertDetail icon={CameraIcon} label="Câmera" value={event.camera_name || event.camera_external_id} />
            <AlertDetail icon={Gauge} label="Confiança" value={formatConfidence(event.confidence)} />
            <AlertDetail icon={AlertTriangle} label="Evento" value={event.event_type} />
            <AlertDetail label="Modelo" value={event.model_alias || "--"} />
            <AlertDetail label="Arquivo" value={event.snapshot_filename || "--"} />
          </div>
        </div>
      </div>
    </div>
  );
}

function SnapshotImage({
  alt,
  className,
  fallbackClassName,
  publicSnapshotPath,
  snapshotPath,
}: {
  alt: string;
  className: string;
  fallbackClassName: string;
  publicSnapshotPath?: string;
  snapshotPath?: string;
}) {
  const [objectUrl, setObjectUrl] = useState("");
  const resolvedSnapshotPath = publicSnapshotPath || snapshotPath;
  const [state, setState] = useState<"empty" | "failed" | "loading" | "ready">(
    resolvedSnapshotPath ? "loading" : "empty",
  );

  useEffect(() => {
    let active = true;
    let createdUrl = "";

    if (!resolvedSnapshotPath) {
      setObjectUrl("");
      setState("empty");
      return () => undefined;
    }

    setState("loading");
    fetchApiAssetBlob(resolvedSnapshotPath)
      .then((blob) => {
        createdUrl = URL.createObjectURL(blob);
        if (!active) {
          URL.revokeObjectURL(createdUrl);
          return;
        }

        setObjectUrl(createdUrl);
        setState("ready");
      })
      .catch(() => {
        if (active) {
          setObjectUrl("");
          setState("failed");
        }
      });

    return () => {
      active = false;
      if (createdUrl) {
        URL.revokeObjectURL(createdUrl);
      }
    };
  }, [resolvedSnapshotPath]);

  if (state === "ready" && objectUrl) {
    return <img alt={alt} className={className} src={objectUrl} />;
  }

  return (
    <div
      className={[
        "grid place-items-center border border-white/[0.08] bg-white/[0.04] text-sm text-slate-400",
        fallbackClassName,
      ].join(" ")}
    >
      <span className="inline-flex items-center gap-2 px-3 text-center">
        <ImageOff className="size-4 text-slate-500" />
        {state === "loading" ? "Carregando imagem" : "Imagem indisponível"}
      </span>
    </div>
  );
}

function AlertDetail({
  icon: Icon,
  label,
  value,
}: {
  icon?: typeof Clock3;
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-lg border border-white/[0.08] bg-white/[0.035] p-4">
      <p className="flex items-center gap-2 text-xs font-semibold uppercase text-slate-500">
        {Icon ? <Icon className="size-4 text-cyan-200" /> : null}
        {label}
      </p>
      <p className="mt-2 break-words text-sm leading-6 text-slate-200">{value}</p>
    </div>
  );
}

export default HomePage;
