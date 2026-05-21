import type { ReactNode } from "react";

interface PanelProps {
  children: ReactNode;
  description?: string;
  title: string;
}

function Panel({ children, description, title }: PanelProps) {
  return (
    <section className="surface rounded-lg p-5 sm:p-6">
      <div className="mb-5">
        <h2 className="font-display text-2xl text-white">{title}</h2>
        {description ? (
          <p className="mt-2 text-sm leading-6 text-slate-400">{description}</p>
        ) : null}
      </div>
      {children}
    </section>
  );
}

export default Panel;
