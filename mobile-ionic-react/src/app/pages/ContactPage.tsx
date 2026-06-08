import { FormEvent, useState } from "react";
import {
  IonButton,
  IonCard,
  IonCardContent,
  IonContent,
  IonIcon,
  IonInput,
  IonItem,
  IonPage,
  IonTextarea,
  IonText,
} from "@ionic/react";
import { mailOutline, personOutline, sendOutline } from "ionicons/icons";
import FeedbackMessage from "../components/FeedbackMessage";
import PageToolbar from "../components/PageToolbar";

const emptyContactForm = {
  email: "",
  message: "",
  name: "",
};

function ContactPage() {
  const [form, setForm] = useState(emptyContactForm);
  const [sent, setSent] = useState(false);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSent(true);
    setForm(emptyContactForm);
  }

  return (
    <IonPage>
      <PageToolbar title="Contato" />
      <IonContent fullscreen className="app-content">
        <section className="page-section">
          <IonCard className="surface-card">
            <IonCardContent>
              <IonText color="light">
                <h2>Fale com a equipe</h2>
              </IonText>
              <IonText color="medium">
                <p>
                  Canal simples para registrar dúvidas sobre implantação,
                  integração ou uso da API no app mobile.
                </p>
              </IonText>

              <FeedbackMessage
                message={sent ? "Mensagem registrada na interface." : ""}
                tone="success"
              />

              <form className="form-stack" onSubmit={handleSubmit}>
                <IonItem className="form-item">
                  <IonIcon icon={personOutline} slot="start" />
                  <IonInput
                    label="Nome"
                    labelPlacement="stacked"
                    required
                    value={form.name}
                    onIonInput={(event) =>
                      setForm((current) => ({
                        ...current,
                        name: String(event.detail.value ?? ""),
                      }))
                    }
                  />
                </IonItem>
                <IonItem className="form-item">
                  <IonIcon icon={mailOutline} slot="start" />
                  <IonInput
                    label="E-mail"
                    labelPlacement="stacked"
                    required
                    type="email"
                    value={form.email}
                    onIonInput={(event) =>
                      setForm((current) => ({
                        ...current,
                        email: String(event.detail.value ?? ""),
                      }))
                    }
                  />
                </IonItem>
                <IonItem className="form-item">
                  <IonTextarea
                    autoGrow
                    label="Mensagem"
                    labelPlacement="stacked"
                    required
                    rows={5}
                    value={form.message}
                    onIonInput={(event) =>
                      setForm((current) => ({
                        ...current,
                        message: String(event.detail.value ?? ""),
                      }))
                    }
                  />
                </IonItem>
                <IonButton expand="block" type="submit">
                  <IonIcon icon={sendOutline} slot="start" />
                  Enviar
                </IonButton>
              </form>
            </IonCardContent>
          </IonCard>
        </section>
      </IonContent>
    </IonPage>
  );
}

export default ContactPage;
