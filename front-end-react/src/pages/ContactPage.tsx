import { useState } from "react";
import type { ChangeEvent, FormEvent } from "react";
import { Mail, MapPin, MessageSquare, Send, User } from "lucide-react";
import ActionButton from "../components/ui/ActionButton";
import FormField, { TextAreaField } from "../components/ui/FormField";
import PageHeader from "../components/ui/PageHeader";
import Panel from "../components/ui/Panel";

type ContactForm = {
  email: string;
  message: string;
  name: string;
};

const initialForm: ContactForm = {
  email: "",
  message: "",
  name: "",
};

function ContactPage() {
  const [form, setForm] = useState<ContactForm>(initialForm);
  const [sent, setSent] = useState(false);

  function updateField(
    field: keyof ContactForm,
    event: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) {
    setForm((current) => ({
      ...current,
      [field]: event.target.value,
    }));
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSent(true);
    setForm(initialForm);
  }

  return (
    <>
      <PageHeader
        eyebrow="Contato"
        title="Fale com a equipe Stealth Lens."
        description="Use este canal para conversas comerciais, duvidas sobre implantacao ou proximos passos do produto."
      />

      <div className="mx-auto grid max-w-7xl gap-6 px-4 pb-12 sm:px-6 lg:grid-cols-[0.9fr_1.1fr]">
        <Panel title="Canais" description="Informacoes organizadas para contato rapido.">
          <div className="grid gap-4">
            <div className="flex items-start gap-3 border-t border-white/[0.08] pt-4">
              <Mail className="mt-0.5 size-5 text-cyan-200" />
              <div>
                <p className="font-semibold text-white">E-mail</p>
                <a
                  className="mt-1 block text-sm text-slate-400 transition hover:text-cyan-100"
                  href="mailto:contato@scientia-vinces.com"
                >
                  contato@scientia-vinces.com
                </a>
              </div>
            </div>
            <div className="flex items-start gap-3 border-t border-white/[0.08] pt-4">
              <MapPin className="mt-0.5 size-5 text-cyan-200" />
              <div>
                <p className="font-semibold text-white">Atendimento</p>
                <p className="mt-1 text-sm text-slate-400">
                  Projetos de monitoramento, integracao e IA aplicada a cameras.
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3 border-t border-white/[0.08] pt-4">
              <MessageSquare className="mt-0.5 size-5 text-cyan-200" />
              <div>
                <p className="font-semibold text-white">Resposta</p>
                <p className="mt-1 text-sm text-slate-400">
                  A equipe pode transformar a mensagem em briefing comercial ou
                  tecnico.
                </p>
              </div>
            </div>
          </div>
        </Panel>

        <Panel title="Mensagem" description="Conte para a equipe qual cenario voce quer monitorar.">
          {sent ? (
            <p className="mb-4 rounded-lg border border-emerald-300/20 bg-emerald-300/10 px-4 py-3 text-sm text-emerald-100">
              Mensagem registrada na interface.
            </p>
          ) : null}

          <form className="grid gap-4" onSubmit={handleSubmit}>
            <FormField
              icon={User}
              id="contact-name"
              label="Nome"
              name="name"
              onChange={(event) => updateField("name", event)}
              placeholder="Seu nome"
              required
              value={form.name}
            />
            <FormField
              icon={Mail}
              id="contact-email"
              label="E-mail"
              name="email"
              onChange={(event) => updateField("email", event)}
              placeholder="voce@empresa.com"
              required
              type="email"
              value={form.email}
            />
            <TextAreaField
              id="contact-message"
              label="Mensagem"
              onChange={(event) => updateField("message", event)}
              placeholder="Conte rapidamente o que voce precisa."
              required
              value={form.message}
            />
            <ActionButton icon={Send} type="submit">
              Enviar mensagem
            </ActionButton>
          </form>
        </Panel>
      </div>
    </>
  );
}

export default ContactPage;
