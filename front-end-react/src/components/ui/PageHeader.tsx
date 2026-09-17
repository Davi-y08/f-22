import type { ReactNode } from "react";

interface PageHeaderProps {
  action?: ReactNode;
  eyebrow?: string;
  title: string;
  description: string;
}

function PageHeader({ action, eyebrow, title, description }: PageHeaderProps) {
  return (
    <section className="mx-auto grid max-w-7xl gap-6 px-4 pb-8 pt-10 sm:px-6 sm:pb-10 sm:pt-14 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-end">
      <div className="max-w-3xl">
        {eyebrow ? (
          <span className="mb-4 inline-flex items-center gap-2 rounded-full border border-cyan-700/20 bg-cyan-700/8 px-3 py-1 text-xs font-bold uppercase tracking-wider text-cyan-800 dark:border-cyan-300/20 dark:bg-cyan-300/10 dark:text-cyan-100">
            <span className="size-1.5 rounded-full bg-cyan-500 shadow-[0_0_8px_rgba(6,182,212,0.9)]" />
            {eyebrow}
          </span>
        ) : null}
        <h1 className="font-display text-4xl leading-tight text-slate-900 sm:text-5xl dark:text-white">
          {title}
        </h1>
        <p className="mt-4 text-base leading-8 text-slate-600 sm:text-lg dark:text-slate-400">
          {description}
        </p>
      </div>
      {action ? <div className="lg:justify-self-end">{action}</div> : null}
    </section>
  );
}

export default PageHeader;
