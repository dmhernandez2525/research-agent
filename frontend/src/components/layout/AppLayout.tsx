import { NavLink, Outlet } from "react-router-dom";

import { setApiKey } from "../../api/client";
import { useLocalStorage } from "../../hooks/useLocalStorage";

const NAV_LINKS = [
  { to: "/", label: "Sessions" },
  { to: "/analytics", label: "Analytics" },
  { to: "/history", label: "History" },
];

function navClass(active: boolean): string {
  if (active) {
    return "rounded-full bg-accent px-4 py-2 text-white";
  }
  return "rounded-full px-4 py-2 text-ink/70 transition hover:bg-white/60 hover:text-ink";
}

export function AppLayout(): JSX.Element {
  const [apiKey, setApiKeyValue] = useLocalStorage<string>("ra_api_key", "");

  const handleApiKeyChange = (value: string) => {
    setApiKeyValue(value);
    setApiKey(value);
  };

  return (
    <div className="mx-auto flex min-h-screen w-full max-w-7xl flex-col px-4 py-6 md:px-8">
      <header className="no-print mb-6 rounded-3xl border border-white/40 bg-white/70 p-4 shadow-soft backdrop-blur md:flex md:items-center md:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-ink/50">Research Agent</p>
          <h1 className="font-serif text-2xl text-ink">Session Control Center</h1>
        </div>
        <div className="mt-4 flex flex-col gap-3 md:mt-0 md:items-end">
          <nav className="flex flex-wrap gap-2">
            {NAV_LINKS.map((item) => (
              <NavLink key={item.to} to={item.to} className={({ isActive }) => navClass(isActive)}>
                {item.label}
              </NavLink>
            ))}
          </nav>
          <label className="flex items-center gap-2 text-xs text-ink/70">
            API Key
            <input
              type="password"
              value={apiKey}
              onChange={(event) => handleApiKeyChange(event.target.value)}
              placeholder="ra_..."
              className="w-56 rounded-full border border-slate-300 bg-white px-3 py-1 text-sm text-ink outline-none focus:border-accent"
            />
          </label>
        </div>
      </header>

      <main className="flex-1 animate-fade-up">
        <Outlet />
      </main>
    </div>
  );
}
