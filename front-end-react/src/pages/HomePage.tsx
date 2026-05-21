import { useEffect, useMemo, useState } from "react";
import type { ChangeEvent, FormEvent } from "react";
import {
  Ban,
  Camera as CameraIcon,
  CheckCircle2,
  Copy,
  KeyRound,
  MapPin,
  Pencil,
  Plus,
  RefreshCw,
  RotateCcw,
  Save,
  Trash2,
  Video,
} from "lucide-react";
import { Link } from "react-router-dom";
import ActionButton from "../components/ui/ActionButton";
import FormField from "../components/ui/FormField";
import PageHeader from "../components/ui/PageHeader";
import Panel from "../components/ui/Panel";
import StatusBadge from "../components/ui/StatusBadge";
import { API_BASE_URL } from "../config/api";
import { agentKeyApi, authApi, cameraApi, getErrorMessage } from "../lib/apiClient";
import type { AgentAccessKey, UserProfile } from "../lib/apiClient";
import type { Camera, CameraPayload, CameraStatus } from "../types/camera";

const agentKeyStorageKey = "stealth-lens-agent-key";

const initialForm: CameraPayload = {
  location: "",
  name: "",
  status: "unknown",
  url: "",
};

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

function HomePage() {
  const [agentKeyName, setAgentKeyName] = useState("Distribuido Stealth Lens");
  const [agentKeys, setAgentKeys] = useState<AgentAccessKey[]>([]);
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [form, setForm] = useState<CameraPayload>(initialForm);
  const [keyError, setKeyError] = useState("");
  const [keySuccess, setKeySuccess] = useState("");
  const [latestAgentKey, setLatestAgentKey] = useState(
    () => localStorage.getItem(agentKeyStorageKey) ?? "",
  );
  const [loading, setLoading] = useState(false);
  const [loadingKeys, setLoadingKeys] = useState(false);
  const [savingKey, setSavingKey] = useState(false);
  const [saving, setSaving] = useState(false);
  const [success, setSuccess] = useState("");
  const [user, setUser] = useState<UserProfile | null>(null);

  const totals = useMemo(
    () => ({
      all: cameras.length,
      offline: cameras.filter((camera) => camera.status === "offline").length,
      online: cameras.filter((camera) => camera.status === "online").length,
    }),
    [cameras],
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
      await loadAgentKeys(true);
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

    try {
      if (editingId) {
        await cameraApi.update(editingId, form);
        setSuccess("Camera atualizada com sucesso.");
      } else {
        await cameraApi.create(form);
        setSuccess("Camera cadastrada com sucesso.");
      }

      resetForm();
      await loadCameras();
    } catch (submitError) {
      setError(getErrorMessage(submitError));
    } finally {
      setSaving(false);
    }
  }

  async function deleteCamera(camera: Camera) {
    const shouldDelete = window.confirm(`Excluir a camera "${camera.name}"?`);
    if (!shouldDelete) return;

    setError("");
    setSuccess("");

    try {
      await cameraApi.delete(camera.id);
      setSuccess("Camera excluida.");
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
      setKeyError("Nao foi possivel copiar automaticamente.");
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
        eyebrow="Home"
        title="Painel de cameras"
        description="Gerencie cameras, pontos de monitoramento e estados operacionais em uma tela direta."
        action={
          <div className="flex flex-col gap-3 sm:flex-row">
            <ActionButton
              disabled={loading}
              icon={RefreshCw}
              onClick={() => void loadCameras()}
              variant="secondary"
            >
              Atualizar
            </ActionButton>
            <Link
              className="inline-flex min-h-11 items-center justify-center gap-2 rounded-lg border border-white/10 bg-white/[0.03] px-4 py-2.5 text-sm font-semibold text-slate-100 transition hover:border-cyan-300/28 hover:bg-white/[0.06]"
              to="/login"
            >
              <CheckCircle2 className="size-4" />
              {user ? user.name : "Entrar"}
            </Link>
          </div>
        }
      />

      <div className="mx-auto grid max-w-7xl gap-6 px-4 pb-12 sm:px-6">
        <section className="grid gap-4 md:grid-cols-3">
          {[
            { label: "Total", value: totals.all, detail: "cameras cadastradas" },
            { label: "Online", value: totals.online, detail: "pontos ativos" },
            { label: "Offline", value: totals.offline, detail: "precisam atencao" },
          ].map((item) => (
            <article className="surface rounded-lg p-5" key={item.label}>
              <p className="text-sm font-semibold text-slate-400">{item.label}</p>
              <strong className="mt-2 block font-display text-4xl text-white">
                {item.value}
              </strong>
              <p className="mt-2 text-sm text-slate-400">{item.detail}</p>
            </article>
          ))}
        </section>

        {(error || success) && (
          <div
            className={[
              "rounded-lg border px-4 py-3 text-sm",
              error
                ? "border-red-300/20 bg-red-300/10 text-red-100"
                : "border-emerald-300/20 bg-emerald-300/10 text-emerald-100",
            ].join(" ")}
          >
            {error || success}
          </div>
        )}

        <section className="grid gap-6 xl:grid-cols-[0.9fr_1.1fr]">
          <Panel
            title="Chave do distribuido"
            description="Crie a chave usada pelo app local para sincronizar cameras e eventos."
          >
            {(keyError || keySuccess) && (
              <div
                className={[
                  "mb-4 rounded-lg border px-4 py-3 text-sm",
                  keyError
                    ? "border-red-300/20 bg-red-300/10 text-red-100"
                    : "border-emerald-300/20 bg-emerald-300/10 text-emerald-100",
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
                placeholder="Distribuido da recepcao"
                required
                value={agentKeyName}
              />

              {latestAgentKey ? (
                <div className="rounded-lg border border-cyan-300/20 bg-cyan-300/8 p-4">
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                    <div className="min-w-0">
                      <p className="text-sm font-semibold text-cyan-100">
                        Chave gerada
                      </p>
                      <p className="mt-2 break-all font-mono text-xs leading-6 text-slate-200">
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
            description="A chave completa aparece somente quando e criada; depois, a lista mostra apenas o prefixo."
          >
            <div className="overflow-hidden rounded-lg border border-white/[0.08]">
              <div className="grid grid-cols-[1fr_auto] gap-4 border-b border-white/[0.08] bg-white/[0.04] px-4 py-3 text-sm font-semibold text-slate-300">
                <span>Chave</span>
                <span>Acoes</span>
              </div>

              {loadingKeys ? (
                <p className="px-4 py-6 text-sm text-slate-400">Carregando chaves...</p>
              ) : agentKeys.length === 0 ? (
                <p className="px-4 py-6 text-sm text-slate-400">
                  Nenhuma chave cadastrada ainda.
                </p>
              ) : (
                <div className="divide-y divide-white/[0.08]">
                  {agentKeys.map((key) => (
                    <div
                      className="grid gap-4 px-4 py-4 md:grid-cols-[minmax(0,1fr)_auto] md:items-center"
                      key={key.id}
                    >
                      <div className="min-w-0">
                        <div className="flex flex-wrap items-center gap-2">
                          <h3 className="font-semibold text-white">{key.name}</h3>
                          <StatusBadge tone={key.revoked_at ? "danger" : "success"}>
                            {key.revoked_at ? "Revogada" : "Ativa"}
                          </StatusBadge>
                        </div>
                        <p className="mt-1 text-sm text-slate-400">
                          Prefixo: {key.key_prefix}
                        </p>
                        {key.last_used_at ? (
                          <p className="mt-1 text-xs text-slate-500">
                            Ultimo uso: {new Date(key.last_used_at).toLocaleString()}
                          </p>
                        ) : null}
                      </div>

                      <div className="flex gap-2 md:justify-end">
                        {!key.revoked_at ? (
                          <button
                            className="inline-flex size-10 items-center justify-center rounded-lg border border-red-300/20 bg-red-300/10 text-red-100 transition hover:border-red-300/38"
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
            title={editingId ? "Editar camera" : "Nova camera"}
            description={`API configurada: ${API_BASE_URL}`}
          >
            <form className="grid gap-4" onSubmit={handleSubmit}>
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
                placeholder="Portao, recepcao, estacionamento..."
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
              <label className="grid gap-2 text-sm font-medium text-slate-200" htmlFor="camera-status">
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
                  {saving ? "Salvando..." : editingId ? "Salvar edicao" : "Cadastrar"}
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
            title="Cameras cadastradas"
            description="Registros ativos da operacao, com status, localizacao e origem de sincronizacao quando houver."
          >
            <div className="overflow-hidden rounded-lg border border-white/[0.08]">
              <div className="grid grid-cols-[1fr_auto] gap-4 border-b border-white/[0.08] bg-white/[0.04] px-4 py-3 text-sm font-semibold text-slate-300">
                <span>Camera</span>
                <span>Acoes</span>
              </div>

              {loading ? (
                <p className="px-4 py-6 text-sm text-slate-400">Carregando cameras...</p>
              ) : cameras.length === 0 ? (
                <p className="px-4 py-6 text-sm text-slate-400">
                  Nenhuma camera carregada ainda.
                </p>
              ) : (
                <div className="divide-y divide-white/[0.08]">
                  {cameras.map((camera) => (
                    <div
                      className="grid gap-4 px-4 py-4 md:grid-cols-[minmax(0,1fr)_auto] md:items-center"
                      key={camera.id}
                    >
                      <div className="min-w-0">
                        <div className="flex flex-wrap items-center gap-2">
                          <h3 className="font-semibold text-white">{camera.name}</h3>
                          <StatusBadge tone={getStatusTone(camera.status)}>
                            {formatStatus(camera.status)}
                          </StatusBadge>
                        </div>
                        <p className="mt-1 text-sm text-slate-400">{camera.location}</p>
                        <p className="mt-1 break-all text-xs text-cyan-100/80">
                          {camera.url}
                        </p>
                        {camera.external_id ? (
                          <p className="mt-1 text-xs text-emerald-100/80">
                            Sync: {camera.agent_id}/{camera.external_id}
                          </p>
                        ) : null}
                      </div>

                      <div className="flex gap-2 md:justify-end">
                        <button
                          className="inline-flex size-10 items-center justify-center rounded-lg border border-white/10 bg-white/[0.03] text-slate-200 transition hover:border-cyan-300/28 hover:text-cyan-100"
                          type="button"
                          aria-label={`Editar ${camera.name}`}
                          title="Editar"
                          onClick={() => selectCamera(camera)}
                        >
                          <Pencil className="size-4" />
                        </button>
                        <button
                          className="inline-flex size-10 items-center justify-center rounded-lg border border-red-300/20 bg-red-300/10 text-red-100 transition hover:border-red-300/38"
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
    </>
  );
}

export default HomePage;
