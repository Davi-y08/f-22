import { useEffect, useState } from "react";
import type { ChangeEvent, FormEvent } from "react";
import { Eye, EyeOff, KeyRound, Mail, ShieldCheck, User } from "lucide-react";
import { NavLink, useNavigate } from "react-router-dom";
import FormField from "../components/ui/FormField";
import ActionButton from "../components/ui/ActionButton";
import StatusBadge from "../components/ui/StatusBadge";
import { authApi, getErrorMessage } from "../lib/apiClient";

type AuthMode = "login" | "cadastro";

interface LoginCadastroPageProps {
  initialMode: AuthMode;
}

type LoginForm = {
  email: string;
  password: string;
};

type SignupForm = {
  confirmPassword: string;
  email: string;
  name: string;
  password: string;
};

const emptyLoginForm: LoginForm = {
  email: "",
  password: "",
};

const emptySignupForm: SignupForm = {
  confirmPassword: "",
  email: "",
  name: "",
  password: "",
};

function LoginCadastroPage({ initialMode }: LoginCadastroPageProps) {
  const navigate = useNavigate();
  const [mode, setMode] = useState<AuthMode>(initialMode);
  const [loginForm, setLoginForm] = useState<LoginForm>(emptyLoginForm);
  const [signupForm, setSignupForm] = useState<SignupForm>(emptySignupForm);
  const [showPassword, setShowPassword] = useState(false);
  const [feedback, setFeedback] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    setMode(initialMode);
    setError("");
    setFeedback("");
  }, [initialMode]);

  function updateLoginField(event: ChangeEvent<HTMLInputElement>) {
    const { name, value } = event.target;
    setLoginForm((current) => ({ ...current, [name]: value }));
  }

  function updateSignupField(event: ChangeEvent<HTMLInputElement>) {
    const { name, value } = event.target;
    setSignupForm((current) => ({ ...current, [name]: value }));
  }

  async function handleLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setError("");
    setFeedback("");

    try {
      await authApi.login(loginForm);
      setFeedback("Login realizado. Redirecionando para a Home...");
      navigate("/");
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
      setFeedback("Cadastro criado. Agora voce pode fazer login.");
      setMode("login");
      navigate("/login");
    } catch (signupError) {
      setError(getErrorMessage(signupError));
    } finally {
      setSubmitting(false);
    }
  }

  function passwordToggle() {
    return (
      <button
        className="inline-flex size-8 items-center justify-center rounded-lg text-slate-400 transition hover:bg-white/[0.06] hover:text-cyan-200"
        type="button"
        aria-label={showPassword ? "Ocultar senha" : "Mostrar senha"}
        title={showPassword ? "Ocultar senha" : "Mostrar senha"}
        onClick={() => setShowPassword((current) => !current)}
      >
        {showPassword ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
      </button>
    );
  }

  return (
    <div className="mx-auto grid max-w-7xl gap-6 px-4 py-10 sm:px-6 sm:py-14 lg:grid-cols-[0.92fr_1.08fr] lg:items-start">
      <section className="surface rounded-lg p-6 sm:p-8">
        <div className="flex items-center gap-4">
          <img
            className="size-14 rounded-lg object-cover ring-1 ring-white/10"
            src="/assets/stealth-lens-logo.jpg"
            alt="Stealth Lens"
          />
          <div>
            <p className="text-sm font-semibold text-cyan-200">Stealth Lens</p>
            <h1 className="font-display text-3xl text-white sm:text-4xl">
              Acesso simples para uma operacao mais calma.
            </h1>
          </div>
        </div>

        <p className="mt-6 max-w-2xl text-base leading-8 text-slate-400">
          Entre para acompanhar cameras, organizar pontos de monitoramento e
          manter o painel pronto para a rotina da equipe.
        </p>

        <div className="mt-8 grid gap-3">
          {[
            "Acesso centralizado para operadores e equipes.",
            "Dados de cameras e sessao em um fluxo mais leve.",
            "Entrada protegida para a rotina de monitoramento.",
          ].map((item) => (
            <div
              className="flex items-start gap-3 border-t border-white/[0.08] pt-3 text-sm leading-6 text-slate-300"
              key={item}
            >
              <ShieldCheck className="mt-0.5 size-4 shrink-0 text-cyan-200" />
              {item}
            </div>
          ))}
        </div>
      </section>

      <section className="surface rounded-lg p-5 sm:p-6">
        <div className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <StatusBadge tone={mode === "login" ? "success" : "neutral"}>
              {mode === "login" ? "Login" : "Cadastro"}
            </StatusBadge>
            <h2 className="mt-3 font-display text-3xl text-white">
              {mode === "login" ? "Bem-vindo de volta" : "Crie sua conta"}
            </h2>
          </div>
          <div className="grid grid-cols-2 rounded-lg border border-white/10 bg-white/[0.03] p-1">
            <NavLink
              className={({ isActive }) =>
                [
                  "rounded-md px-4 py-2 text-center text-sm font-semibold transition",
                  isActive ? "bg-cyan-300 text-slate-950" : "text-slate-300",
                ].join(" ")
              }
              to="/login"
            >
              Login
            </NavLink>
            <NavLink
              className={({ isActive }) =>
                [
                  "rounded-md px-4 py-2 text-center text-sm font-semibold transition",
                  isActive ? "bg-cyan-300 text-slate-950" : "text-slate-300",
                ].join(" ")
              }
              to="/cadastro"
            >
              Cadastro
            </NavLink>
          </div>
        </div>

        {error ? (
          <p className="mb-4 rounded-lg border border-red-300/20 bg-red-300/10 px-4 py-3 text-sm text-red-100">
            {error}
          </p>
        ) : null}

        {feedback ? (
          <p className="mb-4 rounded-lg border border-emerald-300/20 bg-emerald-300/10 px-4 py-3 text-sm text-emerald-100">
            {feedback}
          </p>
        ) : null}

        {mode === "login" ? (
          <form className="grid gap-5" onSubmit={handleLogin}>
            <FormField
              autoComplete="email"
              icon={Mail}
              id="login-email"
              label="E-mail"
              name="email"
              onChange={updateLoginField}
              placeholder="operador@empresa.com"
              required
              type="email"
              value={loginForm.email}
            />
            <FormField
              autoComplete="current-password"
              icon={KeyRound}
              id="login-password"
              label="Senha"
              name="password"
              onChange={updateLoginField}
              placeholder="Digite sua senha"
              required
              trailingAction={passwordToggle()}
              type={showPassword ? "text" : "password"}
              value={loginForm.password}
            />
            <ActionButton disabled={submitting} icon={ShieldCheck} type="submit">
              {submitting ? "Entrando..." : "Entrar"}
            </ActionButton>
          </form>
        ) : (
          <form className="grid gap-5" onSubmit={handleSignup}>
            <FormField
              autoComplete="name"
              icon={User}
              id="signup-name"
              label="Nome"
              name="name"
              onChange={updateSignupField}
              placeholder="Nome da pessoa responsável"
              required
              value={signupForm.name}
            />
            <FormField
              autoComplete="email"
              icon={Mail}
              id="signup-email"
              label="E-mail"
              name="email"
              onChange={updateSignupField}
              placeholder="time@empresa.com"
              required
              type="email"
              value={signupForm.email}
            />
            <div className="grid gap-5 sm:grid-cols-2">
              <FormField
                autoComplete="new-password"
                icon={KeyRound}
                id="signup-password"
                label="Senha"
                name="password"
                onChange={updateSignupField}
                placeholder="Crie uma senha"
                required
                trailingAction={passwordToggle()}
                type={showPassword ? "text" : "password"}
                value={signupForm.password}
              />
              <FormField
                autoComplete="new-password"
                icon={ShieldCheck}
                id="signup-confirm"
                label="Confirmar senha"
                name="confirmPassword"
                onChange={updateSignupField}
                placeholder="Repita a senha"
                required
                type={showPassword ? "text" : "password"}
                value={signupForm.confirmPassword}
              />
            </div>
            <ActionButton disabled={submitting} icon={User} type="submit">
              {submitting ? "Criando..." : "Criar conta"}
            </ActionButton>
          </form>
        )}
      </section>
    </div>
  );
}

export default LoginCadastroPage;
