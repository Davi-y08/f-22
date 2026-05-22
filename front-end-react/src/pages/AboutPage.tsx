import { Activity, Boxes, HeartHandshake, ShieldCheck } from "lucide-react";
import PageHeader from "../components/ui/PageHeader";
import Panel from "../components/ui/Panel";

const values = [
  {
    icon: ShieldCheck,
    title: "Segurança no centro",
    text: "A interface foi pensada para equipes que precisam agir com clareza, sem excesso visual competindo com informações importantes.",
  },
  {
    icon: Activity,
    title: "Rotina em tempo real",
    text: "O produto conversa com o contexto de câmeras ativas, estados operacionais e sinais que pedem acompanhamento constante.",
  },
  {
    icon: HeartHandshake,
    title: "Mais humano",
    text: "A experiência prioriza textos diretos, controles previsíveis e uma leitura mais tranquila para a equipe.",
  },
  {
    icon: Boxes,
    title: "Base modular",
    text: "A estrutura continua separada em componentes, páginas, rotas e camada de API para crescer junto com o projeto.",
  },
];

function AboutPage() {
  return (
    <>
      <PageHeader
        eyebrow="Sobre"
        title="Uma plataforma de visão computacional com cara de produto."
        description="Stealth Lens organiza câmeras e sinais de vigilância em uma experiência escura, discreta e mais fácil de operar no dia a dia."
      />

      <div className="mx-auto grid max-w-7xl gap-6 px-4 pb-12 sm:px-6 lg:grid-cols-[0.95fr_1.05fr]">
        <Panel title="Identidade" description="Visão do produto e sua proposta operacional.">
          <div className="flex flex-col gap-5 text-base leading-8 text-slate-400">
            <p>
              Stealth Lens nasce para ambientes que já convivem com câmeras,
              alertas e decisões rápidas. A proposta é aproximar esses sinais
              de uma rotina clara, confiável e fácil de acompanhar.
            </p>
            <p>
              A plataforma combina organização de pontos de monitoramento com
              uma base preparada para visão computacional, automação e expansão
              por cenários.
            </p>
          </div>
        </Panel>

        <section className="grid gap-4 sm:grid-cols-2">
          {values.map((item) => {
            const Icon = item.icon;

            return (
              <article className="surface rounded-lg p-5" key={item.title}>
                <Icon className="size-5 text-cyan-200" />
                <h2 className="mt-4 font-display text-xl text-white">
                  {item.title}
                </h2>
                <p className="mt-3 text-sm leading-7 text-slate-400">
                  {item.text}
                </p>
              </article>
            );
          })}
        </section>
      </div>
    </>
  );
}

export default AboutPage;
