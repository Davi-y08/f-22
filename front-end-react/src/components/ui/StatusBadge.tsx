interface StatusBadgeProps {
  tone?: "success" | "warning" | "neutral" | "danger";
  children: string;
}

const toneClasses = {
  success: "border-emerald-300/22 bg-emerald-300/10 text-emerald-100",
  warning: "border-amber-300/22 bg-amber-300/10 text-amber-100",
  neutral: "border-slate-300/16 bg-white/[0.04] text-slate-200",
  danger: "border-red-300/22 bg-red-300/10 text-red-100",
};

function StatusBadge({ children, tone = "neutral" }: StatusBadgeProps) {
  return (
    <span
      className={[
        "inline-flex items-center rounded-lg border px-2.5 py-1 text-xs font-semibold",
        toneClasses[tone],
      ].join(" ")}
    >
      {children}
    </span>
  );
}

export default StatusBadge;
