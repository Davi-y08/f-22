import { FormEvent, useEffect, useMemo, useState } from "react";
import {
  IonButton,
  IonCard,
  IonCardContent,
  IonChip,
  IonContent,
  IonIcon,
  IonInput,
  IonItem,
  IonLabel,
  IonList,
  IonLoading,
  IonNote,
  IonPage,
  IonSearchbar,
  IonSelect,
  IonSelectOption,
  IonText,
} from "@ionic/react";
import {
  addOutline,
  cameraOutline,
  checkmarkCircleOutline,
  closeCircleOutline,
  copyOutline,
  createOutline,
  keyOutline,
  locationOutline,
  refreshOutline,
  saveOutline,
  trashOutline,
} from "ionicons/icons";
import BrandHeader from "../components/BrandHeader";
import FeedbackMessage from "../components/FeedbackMessage";
import MetricCard from "../components/MetricCard";
import PageToolbar from "../components/PageToolbar";
import type { AgentAccessKey } from "../models/agent-key.model";
import type { UserProfile } from "../models/auth.model";
import type { Camera, CameraPayload, CameraStatus } from "../models/camera.model";
import {
  agentKeyApi,
  API_BASE_URL,
  cameraApi,
  DEFAULT_API_BASE_URL,
  getErrorMessage,
  authApi,
} from "../services/api.service";
import {
  cameraUsesDefaultUrl,
  emptyCameraForm,
  normalizeCameraPayload,
  validateCameraForm,
} from "../services/camera-form.service";

const agentKeyStorageKey = "stealth-lens-mobile-agent-key";

function formatStatus(status: string) {
  const labels: Record<string, string> = {
    offline: "Offline",
    online: "Online",
    unknown: "Sem sinal",
  };

  return labels[status] ?? status;
}

function getStatusClass(status: string) {
  if (status === "online") return "status-online";
  if (status === "offline") return "status-offline";
  return "status-unknown";
}

function HomePage() {
  const [agentKeyName, setAgentKeyName] = useState("Distribuído Stealth Lens");
  const [agentKeys, setAgentKeys] = useState<AgentAccessKey[]>([]);
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [form, setForm] = useState<CameraPayload>(emptyCameraForm);
  const [keyError, setKeyError] = useState("");
  const [keySuccess, setKeySuccess] = useState("");
  const [latestAgentKey, setLatestAgentKey] = useState(
    () => localStorage.getItem(agentKeyStorageKey) ?? "",
  );
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [savingKey, setSavingKey] = useState(false);
  const [search, setSearch] = useState("");
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

  const filteredCameras = useMemo(() => {
    const term = search.trim().toLowerCase();

    if (!term) {
      return cameras;
    }

    return cameras.filter((camera) =>
      [camera.name, camera.location, camera.status, camera.url]
        .join(" ")
        .toLowerCase()
        .includes(term),
    );
  }, [cameras, search]);

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

  async function loadAgentKeys(quiet = false) {
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
    setForm(emptyCameraForm);
  }

  function selectCamera(camera: Camera) {
    setEditingId(camera.id);
    setForm({
      location: camera.location,
      name: camera.name,
      status: (camera.status as CameraStatus) || "unknown",
      url: camera.url,
    });
    window.scrollTo({ behavior: "smooth", top: 0 });
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
      const usedDefaultUrl = cameraUsesDefaultUrl(form, payload);

      if (editingId) {
        await cameraApi.update(editingId, payload);
        setSuccess(
          usedDefaultUrl
            ? `Câmera atualizada. A URL inválida foi trocada por ${DEFAULT_API_BASE_URL}.`
            : "Câmera atualizada com sucesso.",
        );
      } else {
        await cameraApi.create(payload);
        setSuccess(
          usedDefaultUrl
            ? `Câmera cadastrada. A URL inválida foi trocada por ${DEFAULT_API_BASE_URL}.`
            : "Câmera cadastrada com sucesso.",
        );
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
    const shouldDelete = window.confirm(`Excluir a câmera "${camera.name}"?`);
    if (!shouldDelete) return;

    setError("");
    setSuccess("");

    try {
      await cameraApi.delete(camera.id);
      if (editingId === camera.id) resetForm();
      setSuccess("Câmera excluída.");
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
      setKeySuccess("Chave criada. Guarde a chave completa antes de sair.");
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

  return (
    <IonPage>
      <PageToolbar title="Stealth Lens" />
      <IonContent fullscreen className="app-content">
        <section className="page-section">
          <BrandHeader subtitle="Monitoramento de câmeras com IA" />

          <IonCard className="surface-card hero-card">
            <IonCardContent>
              <IonText color="light">
                <h2>App mobile para consumir a API do projeto.</h2>
              </IonText>
              <IonText color="medium">
                <p>
                  Integrantes: Caio Yudi, Daniel Rocha, Elton Davi, Igor Lima e
                  Levi Rodrigues. A proposta é cadastrar, consultar e acompanhar
                  câmeras conectadas ao sistema.
                </p>
              </IonText>
              <IonNote>API configurada: {API_BASE_URL}</IonNote>
              {user ? (
                <IonChip className="user-chip">
                  <IonIcon icon={checkmarkCircleOutline} />
                  <IonLabel>{user.name}</IonLabel>
                </IonChip>
              ) : (
                <IonChip className="user-chip muted">
                  <IonIcon icon={closeCircleOutline} />
                  <IonLabel>Faça login para gerenciar</IonLabel>
                </IonChip>
              )}
            </IonCardContent>
          </IonCard>

          <div className="metrics-grid">
            <MetricCard icon={cameraOutline} label="Câmeras" value={totals.all} />
            <MetricCard icon={checkmarkCircleOutline} label="Online" value={totals.online} />
            <MetricCard icon={closeCircleOutline} label="Offline" value={totals.offline} />
          </div>

          <FeedbackMessage message={error} />
          <FeedbackMessage message={success} tone="success" />

          <IonCard className="surface-card">
            <IonCardContent>
              <IonText color="light">
                <h2>{editingId ? "Editar câmera" : "Cadastrar câmera"}</h2>
              </IonText>
              <form className="form-stack" onSubmit={handleSubmit}>
                <IonItem className="form-item">
                  <IonIcon icon={cameraOutline} slot="start" />
                  <IonInput
                    label="Nome"
                    labelPlacement="stacked"
                    required
                    value={form.name}
                    onIonInput={(event) => updateField("name", String(event.detail.value ?? ""))}
                  />
                </IonItem>

                <IonItem className="form-item">
                  <IonIcon icon={locationOutline} slot="start" />
                  <IonInput
                    label="Local"
                    labelPlacement="stacked"
                    required
                    value={form.location}
                    onIonInput={(event) =>
                      updateField("location", String(event.detail.value ?? ""))
                    }
                  />
                </IonItem>

                <IonItem className="form-item">
                  <IonIcon icon={refreshOutline} slot="start" />
                  <IonInput
                    label="URL ou IP"
                    labelPlacement="stacked"
                    placeholder="rtsp://192.168.1.10:554/stream"
                    value={form.url}
                    onIonInput={(event) => updateField("url", String(event.detail.value ?? ""))}
                  />
                </IonItem>

                <IonItem className="form-item">
                  <IonSelect
                    interface="popover"
                    label="Status"
                    labelPlacement="stacked"
                    value={form.status}
                    onIonChange={(event) => updateField("status", String(event.detail.value))}
                  >
                    <IonSelectOption value="unknown">Sem sinal</IonSelectOption>
                    <IonSelectOption value="online">Online</IonSelectOption>
                    <IonSelectOption value="offline">Offline</IonSelectOption>
                  </IonSelect>
                </IonItem>

                <div className="button-row">
                  <IonButton disabled={saving} expand="block" type="submit">
                    <IonIcon icon={editingId ? saveOutline : addOutline} slot="start" />
                    {saving ? "Salvando..." : editingId ? "Salvar edição" : "Cadastrar"}
                  </IonButton>
                  {editingId ? (
                    <IonButton color="medium" fill="outline" onClick={resetForm} type="button">
                      Cancelar
                    </IonButton>
                  ) : null}
                </div>
              </form>
            </IonCardContent>
          </IonCard>

          <IonCard className="surface-card">
            <IonCardContent>
              <IonText color="light">
                <h2>Chave do distribuído</h2>
              </IonText>
              <IonText color="medium">
                <p>Use esta chave no agente local para sincronizar câmeras e eventos.</p>
              </IonText>

              <FeedbackMessage message={keyError} />
              <FeedbackMessage message={keySuccess} tone="success" />

              <form className="form-stack" onSubmit={createAgentKey}>
                <IonItem className="form-item">
                  <IonIcon icon={keyOutline} slot="start" />
                  <IonInput
                    label="Nome da chave"
                    labelPlacement="stacked"
                    required
                    value={agentKeyName}
                    onIonInput={(event) => setAgentKeyName(String(event.detail.value ?? ""))}
                  />
                </IonItem>

                <IonButton disabled={savingKey} expand="block" type="submit">
                  <IonIcon icon={keyOutline} slot="start" />
                  {savingKey ? "Gerando..." : "Gerar chave"}
                </IonButton>
              </form>

              {latestAgentKey ? (
                <div className="key-box">
                  <span>{latestAgentKey}</span>
                  <IonButton fill="clear" onClick={() => void copyLatestAgentKey()}>
                    <IonIcon icon={copyOutline} slot="icon-only" />
                  </IonButton>
                </div>
              ) : null}

              {agentKeys.length > 0 ? (
                <IonList className="app-list">
                  {agentKeys.map((key) => (
                    <IonItem key={key.id}>
                      <IonLabel>
                        <h3>{key.name}</h3>
                        <p>Prefixo: {key.key_prefix}</p>
                      </IonLabel>
                      <IonChip color={key.revoked_at ? "danger" : "success"} slot="end">
                        {key.revoked_at ? "Revogada" : "Ativa"}
                      </IonChip>
                    </IonItem>
                  ))}
                </IonList>
              ) : null}
            </IonCardContent>
          </IonCard>

          <IonCard className="surface-card">
            <IonCardContent>
              <IonText color="light">
                <h2>Listagem de câmeras</h2>
              </IonText>
              <IonSearchbar
                debounce={250}
                placeholder="Buscar por nome, local ou status"
                value={search}
                onIonInput={(event) => setSearch(String(event.detail.value ?? ""))}
              />
              <IonButton fill="outline" onClick={() => void loadCameras()}>
                <IonIcon icon={refreshOutline} slot="start" />
                Atualizar
              </IonButton>

              {filteredCameras.length === 0 && !loading ? (
                <IonText color="medium">
                  <p className="empty-text">Nenhuma câmera encontrada.</p>
                </IonText>
              ) : (
                <IonList className="app-list camera-list">
                  {filteredCameras.map((camera) => (
                    <IonItem key={camera.id} routerLink={`/camera/${camera.id}`}>
                      <IonIcon icon={cameraOutline} slot="start" />
                      <IonLabel>
                        <h3>{camera.name}</h3>
                        <p>{camera.location}</p>
                        <p className="camera-url">{camera.url}</p>
                      </IonLabel>
                      <IonChip className={getStatusClass(camera.status)} slot="end">
                        {formatStatus(camera.status)}
                      </IonChip>
                      <IonButton
                        fill="clear"
                        slot="end"
                        onClick={(event) => {
                          event.preventDefault();
                          event.stopPropagation();
                          selectCamera(camera);
                        }}
                      >
                        <IonIcon icon={createOutline} slot="icon-only" />
                      </IonButton>
                      <IonButton
                        color="danger"
                        fill="clear"
                        slot="end"
                        onClick={(event) => {
                          event.preventDefault();
                          event.stopPropagation();
                          void deleteCamera(camera);
                        }}
                      >
                        <IonIcon icon={trashOutline} slot="icon-only" />
                      </IonButton>
                    </IonItem>
                  ))}
                </IonList>
              )}
            </IonCardContent>
          </IonCard>
        </section>

        <IonLoading isOpen={loading} message="Carregando câmeras..." />
      </IonContent>
    </IonPage>
  );
}

export default HomePage;
