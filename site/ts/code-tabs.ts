/**
 * Rust / Python code tabs that remember the reader's choice.
 *
 * Quarto renders `::: {.panel-tabset group="language"}` as Bootstrap nav tabs and keeps
 * tabs of one group in sync within a page. This adds persistence across pages: the
 * chosen tab name is stored under `trm.language` in localStorage (wrapped in try/catch:
 * private windows and blocked storage must not break anything) and re-applied on load.
 * Without JavaScript the first tab is shown, as Quarto renders it.
 */

const KEY = "trm.language";

function read(): string | null {
  try {
    return window.localStorage.getItem(KEY);
  } catch {
    return null;
  }
}

function write(value: string): void {
  try {
    window.localStorage.setItem(KEY, value);
  } catch {
    /* storage unavailable: the choice lives for this page only */
  }
}

function tabsets(): HTMLElement[] {
  return Array.from(document.querySelectorAll<HTMLElement>('.panel-tabset[data-group="language"], .panel-tabset[group="language"]'));
}

function tabLinks(set: HTMLElement): HTMLAnchorElement[] {
  return Array.from(set.querySelectorAll<HTMLAnchorElement>('ul.nav-tabs a[role="tab"], ul.nav-tabs a.nav-link'));
}

export function applyChoice(name: string): void {
  for (const set of tabsets()) {
    for (const link of tabLinks(set)) {
      if (link.textContent?.trim().toLowerCase() === name.toLowerCase()) {
        if (!link.classList.contains("active")) link.click();
      }
    }
  }
}

export function init(): void {
  const sets = tabsets();
  if (sets.length === 0) return;
  const saved = read();
  if (saved) applyChoice(saved);
  for (const set of sets) {
    for (const link of tabLinks(set)) {
      link.addEventListener("click", () => {
        const name = link.textContent?.trim();
        if (name) write(name);
      });
    }
  }
}

if (typeof document !== "undefined") {
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
}
