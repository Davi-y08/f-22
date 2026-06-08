import { useEffect, useState } from "react";
import {
  IonCard,
  IonCardContent,
  IonChip,
  IonContent,
  IonIcon,
  IonItem,
  IonLabel,
  IonList,
  IonLoading,
  IonPage,
  IonText,
} from "@ionic/react";
import { cameraOutline, linkOutline, locationOutline, timeOutline } from "ionicons/icons";
import { useParams } from "react-router-dom";
import FeedbackMessage from "../components/FeedbackMessage";
import PageToolbar from "../components/PageToolbar";
import type { Camera } from "../models/camera.model";
import { cameraApi, getErrorMessage } from "../services/api.service";

function formatDate(value?: string) {
  if (!value) {
    return "Não informado";
  }

  return new Date(value).toLocaleString("pt-BR");
}

function CameraDetailsPage() {
  const { id } = useParams<{ id: string }>();
  const [camera, setCamera] = useState<Camera | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    async function loadCamera() {
      setLoading(true);
      setError("");

      try {
        const result = await cameraApi.get(id);
        setCamera(result);
      } catch (loadError) {
        setError(getErrorMessage(loadError));
      } finally {
        setLoading(false);
      }
    }

    void loadCamera();
  }, [id]);

  return (
    <IonPage>
      <PageToolbar backHref="/home" title="Detalhes" />
      <IonContent fullscreen className="app-content">
        <section className="page-section">
          <FeedbackMessage message={error} />

          {camera ? (
            <IonCard className="surface-card">
              <IonCardContent>
                <div className="details-header">
                  <IonIcon icon={cameraOutline} />
                  <div>
                    <IonText color="light">
                      <h2>{camera.name}</h2>
                    </IonText>
                    <IonChip>{camera.status}</IonChip>
                  </div>
                </div>

                <IonList className="app-list details-list">
                  <IonItem>
                    <IonIcon icon={locationOutline} slot="start" />
                    <IonLabel>
                      <h3>Local</h3>
                      <p>{camera.location}</p>
                    </IonLabel>
                  </IonItem>
                  <IonItem>
                    <IonIcon icon={linkOutline} slot="start" />
                    <IonLabel>
                      <h3>URL/IP</h3>
                      <p className="camera-url">{camera.url}</p>
                    </IonLabel>
                  </IonItem>
                  <IonItem>
                    <IonIcon icon={timeOutline} slot="start" />
                    <IonLabel>
                      <h3>Criada em</h3>
                      <p>{formatDate(camera.created_at)}</p>
                    </IonLabel>
                  </IonItem>
                  <IonItem>
                    <IonIcon icon={timeOutline} slot="start" />
                    <IonLabel>
                      <h3>Última atualização</h3>
                      <p>{formatDate(camera.updated_at)}</p>
                    </IonLabel>
                  </IonItem>
                  <IonItem>
                    <IonIcon icon={cameraOutline} slot="start" />
                    <IonLabel>
                      <h3>Sincronização</h3>
                      <p>
                        {camera.external_id
                          ? `${camera.agent_id ?? "agente"} / ${camera.external_id}`
                          : "Câmera cadastrada pelo app"}
                      </p>
                    </IonLabel>
                  </IonItem>
                </IonList>
              </IonCardContent>
            </IonCard>
          ) : !loading ? (
            <IonText color="medium">
              <p className="empty-text">Nenhuma câmera carregada.</p>
            </IonText>
          ) : null}
        </section>

        <IonLoading isOpen={loading} message="Carregando detalhes..." />
      </IonContent>
    </IonPage>
  );
}

export default CameraDetailsPage;
