import { useState } from "react";
import { Menu, Moon, Sun, X } from "lucide-react";
import { NavLink } from "react-router-dom";
import { useTheme } from "../../hooks/useTheme";

const navLinks = [
  { to: "/", label: "Home" },
  { to: "/sobre", label: "Sobre" },
  { to: "/contato", label: "Contato" },
];

function getDesktopNavClass({ isActive }: { isActive: boolean }) {
  return [
    "px-5 py-2 text-sm font-medium transition-colors duration-200",
    isActive
      ? "text-cyan-800 underline decoration-cyan-400 decoration-2 underline-offset-8 dark:text-cyan-200 dark:decoration-cyan-400"
      : "text-slate-600 hover:text-slate-900 dark:text-slate-300 dark:hover:text-white",
  ].join(" ");
}

function getMobileNavClass({ isActive }: { isActive: boolean }) {
  return [
    "rounded-lg px-4 py-2.5 text-sm font-medium transition",
    isActive
      ? "bg-slate-900/[0.06] text-slate-900 dark:bg-cyan-300/12 dark:text-cyan-100"
      : "text-slate-600 hover:bg-slate-900/[0.04] hover:text-slate-900 dark:text-slate-300 dark:hover:bg-white/[0.05] dark:hover:text-white",
  ].join(" ");
}

function Navbar() {
  const [isOpen, setIsOpen] = useState(false);
  const { theme, toggleTheme } = useTheme();

  return (
    <header className="sticky top-0 z-40 border-b border-slate-900/[0.06] bg-white/80 shadow-[0_1px_0_rgba(2,44,80,0.03),0_8px_24px_rgba(2,44,80,0.05)] backdrop-blur-xl dark:border-white/[0.08] dark:bg-[#04070c]/82 dark:shadow-[0_8px_28px_rgba(0,0,0,0.35)]">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between gap-4 px-4 sm:px-6 lg:px-8">
        <NavLink
          className="group flex min-w-0 items-center gap-2.5"
          to="/"
          aria-label="Stealth Lens home"
          onClick={() => setIsOpen(false)}
        >
          <span className="relative shrink-0 overflow-hidden rounded-xl ring-1 ring-slate-900/10 shadow-[0_8px_20px_rgba(32,33,64,0.2)] dark:ring-white/10 dark:shadow-[0_8px_20px_rgba(0,0,0,0.45)]">
            <img
              className="size-10 object-cover transition-transform duration-300 group-hover:scale-105"
              src="/assets/logo.jpg"
              alt="Stealth Lens"
            />
          </span>
          <span className="min-w-0">
            <span className="block truncate font-display text-lg font-semibold leading-none text-slate-900 dark:text-white">
              Stealth Lens
            </span>
            <span className="mt-1 block truncate text-[10px] font-semibold uppercase tracking-[0.16em] text-cyan-700 dark:text-cyan-300">
              Scientia Vinces
            </span>
          </span>
        </NavLink>

        <nav
          className="hidden items-center gap-2 md:flex"
          aria-label="Principal"
        >
          {navLinks.map((link) => (
            <NavLink
              end={link.to === "/"}
              key={link.to}
              className={getDesktopNavClass}
              to={link.to}
            >
              {link.label}
            </NavLink>
          ))}
        </nav>

        <div className="flex items-center gap-2">
          <button
            className="inline-flex size-10 items-center justify-center rounded-full border border-slate-900/10 bg-white text-slate-600 shadow-sm transition-colors duration-200 hover:bg-slate-900/[0.04] hover:text-slate-900 dark:border-white/10 dark:bg-white/[0.04] dark:text-slate-300 dark:hover:bg-white/[0.08] dark:hover:text-white"
            type="button"
            aria-label={theme === "dark" ? "Ativar tema claro" : "Ativar tema escuro"}
            title={theme === "dark" ? "Ativar tema claro" : "Ativar tema escuro"}
            onClick={toggleTheme}
          >
            {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
          </button>

          <NavLink
            className="inline-flex min-h-10 items-center justify-center rounded-lg bg-[#202140] px-5 py-2 text-sm font-semibold text-white shadow-[0_12px_26px_rgba(32,33,64,0.32)] ring-1 ring-inset ring-white/10 transition-all duration-200 ease-out hover:bg-[#262750] hover:shadow-[0_14px_28px_rgba(32,33,64,0.36)] dark:bg-gradient-to-r dark:from-cyan-300 dark:via-cyan-400 dark:to-blue-500 dark:text-slate-950 dark:shadow-[0_16px_34px_rgba(14,165,233,0.24)] dark:ring-transparent dark:hover:brightness-110 dark:hover:shadow-[0_16px_34px_rgba(14,165,233,0.3)]"
            to="/login"
          >
            Entrar
          </NavLink>
        </div>

        <button
          className="inline-flex size-10 items-center justify-center rounded-full border border-slate-900/10 bg-white text-slate-700 shadow-sm transition hover:border-slate-900/20 hover:bg-slate-900/[0.03] md:hidden dark:border-white/10 dark:bg-white/[0.04] dark:text-slate-200 dark:hover:border-cyan-300/28"
          type="button"
          aria-label={isOpen ? "Fechar menu" : "Abrir menu"}
          aria-expanded={isOpen}
          title={isOpen ? "Fechar menu" : "Abrir menu"}
          onClick={() => setIsOpen((current) => !current)}
        >
          {isOpen ? <X className="size-5" /> : <Menu className="size-5" />}
        </button>
      </div>

      {isOpen ? (
        <div className="absolute inset-x-0 top-full animate-[slide-down_180ms_ease-out] border-b border-slate-900/[0.06] bg-white/95 shadow-[0_24px_48px_rgba(2,44,80,0.12)] backdrop-blur-xl md:hidden dark:border-white/[0.08] dark:bg-[#04070c]/95 dark:shadow-[0_24px_48px_rgba(0,0,0,0.55)]">
          <nav className="mx-auto grid max-w-7xl gap-1 px-4 py-4 sm:px-6" aria-label="Menu mobile">
            {navLinks.map((link) => (
              <NavLink
                end={link.to === "/"}
                key={link.to}
                className={getMobileNavClass}
                to={link.to}
                onClick={() => setIsOpen(false)}
              >
                {link.label}
              </NavLink>
            ))}
            <NavLink
              className="mt-2 inline-flex min-h-11 items-center justify-center rounded-lg bg-[#202140] px-4 py-2.5 text-sm font-semibold text-white shadow-[0_10px_24px_rgba(32,33,64,0.3)] ring-1 ring-inset ring-white/10 transition-all duration-200 ease-out hover:bg-[#262750] dark:bg-gradient-to-r dark:from-cyan-300 dark:via-cyan-400 dark:to-blue-500 dark:text-slate-950 dark:shadow-[0_16px_34px_rgba(14,165,233,0.24)] dark:ring-transparent dark:hover:brightness-110"
              to="/login"
              onClick={() => setIsOpen(false)}
            >
              Entrar
            </NavLink>
          </nav>
        </div>
      ) : null}
    </header>
  );
}

export default Navbar;