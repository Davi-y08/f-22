import { NavLink } from "react-router-dom";
import InstagramIcon from "../ui/InstagramIcon";

const instagramUrl = "https://www.instagram.com/getstealthlens/";

function Footer() {
  return (
    <footer className="relative overflow-hidden border-t border-cyan-950 bg-[#04070c]">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_80%_120%,rgba(34,211,238,0.14),transparent_45%)]" />
      <div className="relative mx-auto flex max-w-7xl flex-col gap-6 px-4 py-10 sm:px-6 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <strong className="font-display text-lg text-white">Stealth Lens</strong>
          <p className="mt-1 text-sm text-slate-400">
            Vigilância inteligente com uma interface mais simples e humana.
          </p>
        </div>
        <div className="flex flex-col gap-5 sm:flex-row sm:items-center">
          <nav className="flex flex-wrap gap-4 text-sm font-medium text-slate-400">
            <NavLink className="transition hover:text-cyan-300" to="/">
              Home
            </NavLink>
            <NavLink className="transition hover:text-cyan-300" to="/sobre">
              Sobre
            </NavLink>
            <NavLink className="transition hover:text-cyan-300" to="/contato">
              Contato
            </NavLink>
          </nav>
          <a
            aria-label="Stealth Lens no Instagram"
            className="inline-flex items-center gap-2 self-start rounded-full border border-white/15 bg-white/[0.04] px-4 py-2 text-sm font-medium text-slate-300 transition hover:border-cyan-300/40 hover:bg-cyan-300/10 hover:text-cyan-200"
            href={instagramUrl}
            rel="noopener noreferrer"
            target="_blank"
          >
            <InstagramIcon className="size-4" />
            @getstealthlens
          </a>
        </div>
      </div>
    </footer>
  );
}

export default Footer;
