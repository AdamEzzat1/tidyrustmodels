/**
 * Client-side filter for the feature matrix and the gap table.
 *
 * For every `.filterable` container the script adds one search box above the tables
 * inside it; typing hides the rows whose text does not contain the query (case-
 * insensitive) and keeps the header row. The tables are static HTML rendered by
 * Quarto, so without JavaScript every row is simply visible.
 */

export function filterRows(table: HTMLTableElement, query: string): number {
  const q = query.trim().toLowerCase();
  let shown = 0;
  for (const row of Array.from(table.tBodies).flatMap((b) => Array.from(b.rows))) {
    const hit = q === "" || (row.textContent ?? "").toLowerCase().includes(q);
    row.hidden = !hit;
    if (hit) shown += 1;
  }
  return shown;
}

export function attach(container: HTMLElement): HTMLInputElement {
  const tables = Array.from(container.querySelectorAll<HTMLTableElement>("table"));
  const box = document.createElement("div");
  box.className = "table-filter";
  const label = document.createElement("label");
  const id = `table-filter-${Math.random().toString(36).slice(2, 8)}`;
  label.htmlFor = id;
  label.textContent = container.dataset.filterLabel ?? "Filter rows";
  const input = document.createElement("input");
  input.type = "search";
  input.id = id;
  input.placeholder = container.dataset.filterPlaceholder ?? "type to filter, e.g. rsample or unsupported";
  input.autocomplete = "off";
  const count = document.createElement("span");
  count.className = "table-filter-count";
  count.setAttribute("aria-live", "polite");
  const total = tables.reduce((n, t) => n + Array.from(t.tBodies).reduce((m, b) => m + b.rows.length, 0), 0);
  const update = (): void => {
    const shown = tables.reduce((n, t) => n + filterRows(t, input.value), 0);
    count.textContent = input.value.trim() === "" ? `${total} rows` : `${shown} of ${total} rows`;
  };
  input.addEventListener("input", update);
  box.append(label, input, count);
  container.prepend(box);
  update();
  return input;
}

export function init(): void {
  for (const c of Array.from(document.querySelectorAll<HTMLElement>(".filterable"))) {
    if (!c.querySelector(".table-filter")) attach(c);
  }
}

if (typeof document !== "undefined") {
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
}
