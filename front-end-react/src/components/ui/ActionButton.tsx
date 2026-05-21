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
    "bg-gradient-to-r from-cyan-300 via-cyan-400 to-blue-500 text-slate-950 shadow-[0_16px_34px_rgba(14,165,233,0.24)] hover:brightness-110",
  secondary:
    "border border-cyan-300/20 bg-cyan-300/10 text-cyan-50 hover:border-cyan-300/38 hover:bg-cyan-300/16",
  danger:
    "border border-red-300/20 bg-red-300/10 text-red-100 hover:border-red-300/36 hover:bg-red-300/16",
  ghost:
    "border border-white/10 bg-white/[0.03] text-slate-100 hover:border-cyan-300/28 hover:bg-white/[0.06]",
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
        "inline-flex min-h-11 items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-45",
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
