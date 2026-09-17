interface StatusBadgeProps {
  tone?: "success" | "warning" | "neutral" | "danger";
  children: string;
}

const toneClasses = {
  success:
    "border-emerald-500/30 bg-emerald-400/12 text-emerald-700 dark:border-emerald-300/22 dark:bg-emerald-300/10 dark:text-emerald-100",
  warning:
    "border-amber-500/30 bg-amber-400/12 text-amber-700 dark:border-amber-300/22 dark:bg-amber-300/10 dark:text-amber-100",
  neutral:
    "border-slate-500/25 bg-slate-900/[0.04] text-slate-600 dark:border-slate-300/16 dark:bg-white/[0.04] dark:text-slate-200",
  danger:
    "border-red-500/30 bg-red-400/12 text-red-700 dark:border-red-300/22 dark:bg-red-300/10 dark:text-red-100",
};

function StatusBadge({ children, tone = "neutral" }: StatusBadgeProps) {
  return (
    <span
      className={[
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-semibold",
        toneClasses[tone],
      ].join(" ")}
    >
      {children}
    </span>
  );
}

export default StatusBadge;
