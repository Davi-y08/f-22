import type { ButtonHTMLAttributes, ReactNode } from "react";
import type { LucideIcon } from "lucide-react";

type ActionButtonVariant = "primary" | "secondary" | "danger" | "ghost";

interface ActionButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  children: ReactNode;
  icon?: LucideIcon;
  variant?: ActionButtonVariant;
}

const variantClasses: Record<ActionButtonVariant, string> = {
  primary:
    "bg-[#202140] text-white shadow-[0_12px_26px_rgba(32,33,64,0.32)] ring-1 ring-inset ring-white/10 hover:bg-[#262750] hover:shadow-[0_14px_28px_rgba(32,33,64,0.36)] dark:bg-gradient-to-r dark:from-cyan-300 dark:via-cyan-400 dark:to-blue-500 dark:text-slate-950 dark:shadow-[0_16px_34px_rgba(14,165,233,0.24)] dark:ring-transparent dark:hover:brightness-110 dark:hover:shadow-[0_16px_34px_rgba(14,165,233,0.3)]",
  secondary:
    "border border-[#202140]/25 bg-[#202140]/[0.06] text-[#1c1d3a] hover:border-[#202140]/35 hover:bg-[#202140]/[0.09] dark:border-cyan-300/20 dark:bg-cyan-300/10 dark:text-cyan-50 dark:hover:border-cyan-300/38 dark:hover:bg-cyan-300/16 dark:hover:text-cyan-50",
  danger:
    "bg-red-500 text-white shadow-[0_10px_22px_rgba(239,68,68,0.28)] hover:brightness-105 dark:border dark:border-red-300/20 dark:bg-red-300/10 dark:text-red-100 dark:shadow-none dark:hover:border-red-300/36 dark:hover:bg-red-300/16 dark:hover:text-red-100",
  ghost:
    "border border-slate-900/15 bg-white text-slate-700 shadow-sm hover:border-[#202140]/30 hover:bg-slate-900/[0.03] hover:text-[#202140] dark:border-white/10 dark:bg-white/[0.03] dark:text-slate-100 dark:shadow-none dark:hover:border-cyan-300/28 dark:hover:bg-white/[0.06] dark:hover:text-slate-100",
};

function ActionButton({
  children,
  icon: Icon,
  variant = "primary",
  className = "",
  type = "button",
  ...props
}: ActionButtonProps) {
  return (
    <button
      className={[
        "inline-flex min-h-11 items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-semibold transition-all duration-200 ease-out disabled:cursor-not-allowed disabled:opacity-45",
        variantClasses[variant],
        className,
      ].join(" ")}
      type={type}
      {...props}
    >
      {Icon ? <Icon className="size-4" /> : null}
      {children}
    </button>
  );
}

export default ActionButton;
