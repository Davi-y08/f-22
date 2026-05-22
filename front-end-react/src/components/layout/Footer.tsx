import { NavLink } from "react-router-dom";

function Footer() {
  return (
    <footer className="border-t border-white/[0.08]">
      <div className="mx-auto flex max-w-7xl flex-col gap-5 px-4 py-8 sm:flex-row sm:items-center sm:justify-between sm:px-6">
        <div>
          <strong className="font-display text-lg text-white">Stealth Lens</strong>
          <p className="mt-1 text-sm text-slate-400">
            Vigilância inteligente com uma interface mais simples e humana.
          </p>
        </div>
        <nav className="flex flex-wrap gap-4 text-sm text-slate-400">
          <NavLink className="transition hover:text-white" to="/">
            Home
          </NavLink>
          <NavLink className="transition hover:text-white" to="/sobre">
            Sobre
          </NavLink>
          <NavLink className="transition hover:text-white" to="/contato">
            Contato
          </NavLink>
        </nav>
      </div>
    </footer>
  );
}

export default Footer;
