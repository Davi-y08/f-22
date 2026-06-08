import {
  IonCard,
  IonCardContent,
  IonContent,
  IonIcon,
  IonItem,
  IonLabel,
  IonList,
  IonPage,
  IonText,
} from "@ionic/react";
import { analyticsOutline, hardwareChipOutline, peopleOutline, shieldCheckmarkOutline } from "ionicons/icons";
import BrandHeader from "../components/BrandHeader";
import PageToolbar from "../components/PageToolbar";

const values = [
  {
    icon: shieldCheckmarkOutline,
    text: "Autenticação integrada com a API hospedada.",
    title: "Segurança",
  },
  {
    icon: hardwareChipOutline,
    text: "Câmeras organizadas para apoiar o agente de detecção por IA.",
    title: "Operação",
  },
  {
    icon: analyticsOutline,
    text: "Listagem, busca, detalhes e status em uma tela responsiva.",
    title: "Consulta",
  },
];

function AboutPage() {
  return (
    <IonPage>
      <PageToolbar title="Sobre" />
      <IonContent fullscreen className="app-content">
        <section className="page-section">
          <BrandHeader subtitle="Projeto mobile em Ionic React" />

          <IonCard className="surface-card">
            <IonCardContent>
              <IonText color="light">
                <h2>Proposta do aplicativo</h2>
              </IonText>
              <IonText color="medium">
                <p>
                  Esta versão mobile consome a API do Stealth Lens para autenticar
                  usuários, cadastrar câmeras, consultar registros e abrir detalhes
                  dos itens selecionados.
                </p>
              </IonText>
            </IonCardContent>
          </IonCard>

          <IonCard className="surface-card">
            <IonCardContent>
              <IonText color="light">
                <h2>Grupo</h2>
              </IonText>
              <IonItem className="info-item">
                <IonIcon icon={peopleOutline} slot="start" />
                <IonLabel>
                  <h3>Integrantes</h3>
                  <p>Caio Yudi, Daniel Rocha, Elton Davi, Igor Lima e Levi Rodrigues</p>
                </IonLabel>
              </IonItem>
            </IonCardContent>
          </IonCard>

          <IonList className="app-list value-list">
            {values.map((item) => (
              <IonItem key={item.title}>
                <IonIcon icon={item.icon} slot="start" />
                <IonLabel>
                  <h3>{item.title}</h3>
                  <p>{item.text}</p>
                </IonLabel>
              </IonItem>
            ))}
          </IonList>
        </section>
      </IonContent>
    </IonPage>
  );
}

export default AboutPage;
