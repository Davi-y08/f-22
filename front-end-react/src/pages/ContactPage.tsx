import { useState } from "react";
import type { ChangeEvent, FormEvent } from "react";
import { Mail, MapPin, MessageSquare, Send, User } from "lucide-react";
import ActionButton from "../components/ui/ActionButton";
import FormField, { TextAreaField } from "../components/ui/FormField";
import InstagramIcon from "../components/ui/InstagramIcon";
import PageHeader from "../components/ui/PageHeader";
import Panel from "../components/ui/Panel";

const instagramUrl = "https://www.instagram.com/getstealthlens/";

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
       
        title="Fale com a equipe Stealth Lens."
        description="Use este canal para conversas comerciais, dúvidas sobre implantação ou próximos passos do produto."
      />

      <div className="mx-auto grid max-w-7xl gap-6 px-4 pb-12 sm:px-6 lg:grid-cols-[0.9fr_1.1fr]">
        <Panel title="Canais">
          <div className="grid gap-4">
            <div className="flex items-start gap-3 border-t border-slate-900/10 pt-4 dark:border-white/[0.08]">
              <Mail className="mt-0.5 size-5 text-cyan-700 dark:text-cyan-200" />
              <div>
                <p className="font-semibold text-slate-900 dark:text-white">E-mail</p>
                <a
                  className="mt-1 block text-sm text-slate-500 transition hover:text-cyan-700 dark:text-slate-400 dark:hover:text-cyan-100"
                  href="mailto:contatostealthlens@gmail.com"
                >
                  contatostealthlens@gmail.com
                </a>
              </div>
            </div>
            <div className="flex items-start gap-3 border-t border-slate-900/10 pt-4 dark:border-white/[0.08]">
              <MapPin className="mt-0.5 size-5 text-cyan-700 dark:text-cyan-200" />
              <div>
                <p className="font-semibold text-slate-900 dark:text-white">Atendimento</p>
                <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
                 Especializado em soluções de monitoramento, automação e IA aplicada a câmeras.
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3 border-t border-slate-900/10 pt-4 dark:border-white/[0.08]">
              <MessageSquare className="mt-0.5 size-5 text-cyan-700 dark:text-cyan-200" />
              <div>
                <p className="font-semibold text-slate-900 dark:text-white">Resposta</p>
                <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
                 Nossa equipe analisa sua mensagem e direciona o retorno para um briefing comercial ou técnico.
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3 border-t border-slate-900/10 pt-4 dark:border-white/[0.08]">
              <InstagramIcon className="mt-0.5 size-5 text-cyan-700 dark:text-cyan-200" />
              <div>
                <p className="font-semibold text-slate-900 dark:text-white">Instagram</p>
                <a
                  className="mt-1 block text-sm text-slate-500 transition hover:text-cyan-700 dark:text-slate-400 dark:hover:text-cyan-100"
                  href={instagramUrl}
                  rel="noopener noreferrer"
                  target="_blank"
                >
                  @getstealthlens
                </a>
              </div>
            </div>
          </div>
        </Panel>

        <Panel title="Mensagem" description="Conte para a equipe qual cenário você quer monitorar.">
          {sent ? (
            <p className="mb-4 rounded-lg border border-emerald-500/30 bg-emerald-400/12 px-4 py-3 text-sm text-emerald-700 dark:border-emerald-300/20 dark:bg-emerald-300/10 dark:text-emerald-100">
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
              placeholder="Conte rapidamente o que você precisa."
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
