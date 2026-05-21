import { useState } from "react";
import { Menu, ShieldCheck, X } from "lucide-react";
import { NavLink } from "react-router-dom";

const navLinks = [
  { to: "/", label: "Home" },
  { to: "/sobre", label: "Sobre" },
  { to: "/contato", label: "Contato" },
];

function getNavClass({ isActive }: { isActive: boolean }) {
  return [
    "rounded-lg px-3 py-2 text-sm font-medium transition",
    isActive
      ? "bg-cyan-300/12 text-cyan-100"
      : "text-slate-300 hover:bg-white/[0.05] hover:text-white",
  ].join(" ");
}

function Navbar() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <header className="sticky top-0 z-40 border-b border-white/[0.08] bg-[#04070c]/82 backdrop-blur-xl">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-4 sm:px-6">
        <NavLink
          className="group flex min-w-0 items-center gap-3"
          to="/"
          aria-label="Stealth Lens home"
          onClick={() => setIsOpen(false)}
        >
          <img
            className="size-11 rounded-lg object-cover ring-1 ring-white/10 shadow-[0_14px_34px_rgba(0,0,0,0.38)]"
            src="/assets/stealth-lens-logo.jpg"
            alt="Stealth Lens"
          />
          <div className="min-w-0">
            <strong className="block truncate font-display text-lg text-white">
              Stealth Lens
            </strong>
            <span className="block truncate text-xs text-slate-400">
              Scientia Vinces
            </span>
          </div>
        </NavLink>

        <nav className="hidden items-center gap-1 md:flex" aria-label="Principal">
          {navLinks.map((link) => (
            <NavLink
              end={link.to === "/"}
              key={link.to}
              className={getNavClass}
              to={link.to}
            >
              {link.label}
            </NavLink>
          ))}
        </nav>

        <div className="hidden items-center gap-2 md:flex">
          <NavLink
            className={({ isActive }) =>
              [
                "inline-flex items-center gap-2 rounded-lg border px-4 py-2.5 text-sm font-semibold transition",
                isActive
                  ? "border-cyan-300/30 bg-cyan-300/14 text-white"
                  : "border-cyan-300/20 bg-cyan-300/10 text-cyan-50 hover:border-cyan-300/38 hover:bg-cyan-300/16",
              ].join(" ")
            }
            to="/login"
          >
            <ShieldCheck className="size-4" />
            Login
          </NavLink>
        </div>

        <button
          className="inline-flex size-10 items-center justify-center rounded-lg border border-white/10 bg-white/[0.04] text-slate-100 transition hover:border-cyan-300/28 md:hidden"
          type="button"
          aria-label={isOpen ? "Fechar menu" : "Abrir menu"}
          title={isOpen ? "Fechar menu" : "Abrir menu"}
          onClick={() => setIsOpen((current) => !current)}
        >
          {isOpen ? <X className="size-5" /> : <Menu className="size-5" />}
        </button>
      </div>

      {isOpen ? (
        <div className="border-t border-white/[0.08] bg-[#04070c]/95 px-4 py-3 md:hidden">
          <nav className="mx-auto grid max-w-7xl gap-1" aria-label="Menu mobile">
            {navLinks.map((link) => (
              <NavLink
                end={link.to === "/"}
                key={link.to}
                className={getNavClass}
                to={link.to}
                onClick={() => setIsOpen(false)}
              >
                {link.label}
              </NavLink>
            ))}
            <NavLink
              className="mt-2 inline-flex items-center justify-center gap-2 rounded-lg border border-cyan-300/20 bg-cyan-300/10 px-4 py-2.5 text-sm font-semibold text-cyan-50"
              to="/login"
              onClick={() => setIsOpen(false)}
            >
              <ShieldCheck className="size-4" />
              Login
            </NavLink>
          </nav>
        </div>
      ) : null}
    </header>
  );
}

export default Navbar;
