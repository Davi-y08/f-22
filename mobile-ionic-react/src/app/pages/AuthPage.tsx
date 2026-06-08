import { FormEvent, useEffect, useState } from "react";
import {
  IonButton,
  IonCard,
  IonCardContent,
  IonContent,
  IonIcon,
  IonInput,
  IonItem,
  IonLabel,
  IonPage,
  IonSegment,
  IonSegmentButton,
  IonText,
} from "@ionic/react";
import { keyOutline, logInOutline, mailOutline, personAddOutline, personOutline } from "ionicons/icons";
import { useHistory } from "react-router-dom";
import BrandHeader from "../components/BrandHeader";
import FeedbackMessage from "../components/FeedbackMessage";
import PageToolbar from "../components/PageToolbar";
import { authApi, getErrorMessage } from "../services/api.service";

type AuthMode = "login" | "cadastro";

type AuthPageProps = {
  initialMode?: AuthMode;
};

const emptyLoginForm = {
  email: "",
  password: "",
};

const emptySignupForm = {
  confirmPassword: "",
  email: "",
  name: "",
  password: "",
};

function AuthPage({ initialMode = "login" }: AuthPageProps) {
  const history = useHistory();
  const [mode, setMode] = useState<AuthMode>(initialMode);
  const [loginForm, setLoginForm] = useState(emptyLoginForm);
  const [signupForm, setSignupForm] = useState(emptySignupForm);
  const [error, setError] = useState("");
  const [feedback, setFeedback] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    setMode(initialMode);
    setError("");
    setFeedback("");
  }, [initialMode]);

  async function handleLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setError("");
    setFeedback("");

    try {
      await authApi.login(loginForm);
      setFeedback("Login realizado. Redirecionando para a Home...");
      history.replace("/home");
    } catch (loginError) {
      setError(getErrorMessage(loginError));
    } finally {
      setSubmitting(false);
    }
  }

  async function handleSignup(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setError("");
    setFeedback("");

    try {
      await authApi.signup({
        confirm_password: signupForm.confirmPassword,
        email: signupForm.email,
        name: signupForm.name,
        password: signupForm.password,
      });
      setFeedback("Cadastro criado. Agora faça login para acessar as câmeras.");
      setMode("login");
      history.replace("/login");
    } catch (signupError) {
      setError(getErrorMessage(signupError));
    } finally {
      setSubmitting(false);
    }
  }

  function goToMode(nextMode: AuthMode) {
    setMode(nextMode);
    setError("");
    setFeedback("");
    history.replace(nextMode === "login" ? "/login" : "/cadastro");
  }

  return (
    <IonPage>
      <PageToolbar title={mode === "login" ? "Login" : "Cadastro"} />
      <IonContent fullscreen className="app-content">
        <section className="page-section auth-section">
          <BrandHeader subtitle="Acesso mobile ao monitoramento" />

          <IonCard className="surface-card">
            <IonCardContent>
              <IonText color="light">
                <h2>{mode === "login" ? "Bem-vindo de volta" : "Crie sua conta"}</h2>
              </IonText>
              <IonText color="medium">
                <p>
                  Use o mesmo acesso da API hospedada para gerenciar câmeras pelo app.
                </p>
              </IonText>

              <IonSegment
                className="auth-segment"
                value={mode}
                onIonChange={(event) => goToMode(event.detail.value as AuthMode)}
              >
                <IonSegmentButton value="login">
                  <IonLabel>Login</IonLabel>
                </IonSegmentButton>
                <IonSegmentButton value="cadastro">
                  <IonLabel>Cadastro</IonLabel>
                </IonSegmentButton>
              </IonSegment>

              <FeedbackMessage message={error} />
              <FeedbackMessage message={feedback} tone="success" />

              {mode === "login" ? (
                <form className="form-stack" onSubmit={handleLogin}>
                  <IonItem className="form-item">
                    <IonIcon icon={mailOutline} slot="start" />
                    <IonInput
                      autocomplete="email"
                      label="E-mail"
                      labelPlacement="stacked"
                      required
                      type="email"
                      value={loginForm.email}
                      onIonInput={(event) =>
                        setLoginForm((current) => ({
                          ...current,
                          email: String(event.detail.value ?? ""),
                        }))
                      }
                    />
                  </IonItem>

                  <IonItem className="form-item">
                    <IonIcon icon={keyOutline} slot="start" />
                    <IonInput
                      autocomplete="current-password"
                      label="Senha"
                      labelPlacement="stacked"
                      required
                      type="password"
                      value={loginForm.password}
                      onIonInput={(event) =>
                        setLoginForm((current) => ({
                          ...current,
                          password: String(event.detail.value ?? ""),
                        }))
                      }
                    />
                  </IonItem>

                  <IonButton disabled={submitting} expand="block" type="submit">
                    <IonIcon icon={logInOutline} slot="start" />
                    {submitting ? "Entrando..." : "Entrar"}
                  </IonButton>
                </form>
              ) : (
                <form className="form-stack" onSubmit={handleSignup}>
                  <IonItem className="form-item">
                    <IonIcon icon={personOutline} slot="start" />
                    <IonInput
                      autocomplete="name"
                      label="Nome"
                      labelPlacement="stacked"
                      required
                      value={signupForm.name}
                      onIonInput={(event) =>
                        setSignupForm((current) => ({
                          ...current,
                          name: String(event.detail.value ?? ""),
                        }))
                      }
                    />
                  </IonItem>

                  <IonItem className="form-item">
                    <IonIcon icon={mailOutline} slot="start" />
                    <IonInput
                      autocomplete="email"
                      label="E-mail"
                      labelPlacement="stacked"
                      required
                      type="email"
                      value={signupForm.email}
                      onIonInput={(event) =>
                        setSignupForm((current) => ({
                          ...current,
                          email: String(event.detail.value ?? ""),
                        }))
                      }
                    />
                  </IonItem>

                  <IonItem className="form-item">
                    <IonIcon icon={keyOutline} slot="start" />
                    <IonInput
                      autocomplete="new-password"
                      label="Senha"
                      labelPlacement="stacked"
                      required
                      type="password"
                      value={signupForm.password}
                      onIonInput={(event) =>
                        setSignupForm((current) => ({
                          ...current,
                          password: String(event.detail.value ?? ""),
                        }))
                      }
                    />
                  </IonItem>

                  <IonItem className="form-item">
                    <IonIcon icon={keyOutline} slot="start" />
                    <IonInput
                      autocomplete="new-password"
                      label="Confirmar senha"
                      labelPlacement="stacked"
                      required
                      type="password"
                      value={signupForm.confirmPassword}
                      onIonInput={(event) =>
                        setSignupForm((current) => ({
                          ...current,
                          confirmPassword: String(event.detail.value ?? ""),
                        }))
                      }
                    />
                  </IonItem>

                  <IonButton disabled={submitting} expand="block" type="submit">
                    <IonIcon icon={personAddOutline} slot="start" />
                    {submitting ? "Criando..." : "Criar conta"}
                  </IonButton>
                </form>
              )}
            </IonCardContent>
          </IonCard>
        </section>
      </IonContent>
    </IonPage>
  );
}

export default AuthPage;
