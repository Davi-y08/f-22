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
          <p className="mb-3 text-sm font-semibold uppercase text-cyan-200">
            {eyebrow}
          </p>
        ) : null}
        <h1 className="font-display text-4xl leading-tight text-white sm:text-5xl">
          {title}
        </h1>
        <p className="mt-4 text-base leading-8 text-slate-400 sm:text-lg">
          {description}
        </p>
      </div>
      {action ? <div className="lg:justify-self-end">{action}</div> : null}
    </section>
  );
}

export default PageHeader;
