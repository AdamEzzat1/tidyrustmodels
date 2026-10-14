/**
 * Benchmark charts from `data/bench.json` (written by site/scripts/build_pages.py from
 * benchmarks/results/<run>/results.csv: medians and quartiles over cold trials).
 *
 * A chart is any element with `data-bench-chart`; attributes select the cells:
 *   data-phase    wf_fit_s | warm_wf_fit_s | cv_s | tune_s | wfs_s | wall_s | peak_wset_mb
 *   data-runs     comma-separated run letters (G, K, I, J, H); default: all
 *   data-threads  1 or 4; default 1
 *   data-tasks    comma-separated task ids; default: every task with that phase
 *   data-models   comma-separated model names to keep (linear_reg, decision_tree, ...)
 *
 * It draws horizontal bars per task and system (R, Rust, Python) on a log10 axis,
 * with the interquartile range as a whisker, the trial count, the median R / Rust
 * ratio computed from the same medians, and a dagger on configurations BENCHMARKS.md
 * labels as disturbed (the note is shown under the chart). Nothing is drawn without
 * JavaScript; the page text and BENCHMARKS.md carry the numbers independently.
 */

interface Stat {
  median: number;
  q1: number;
  q3: number;
  n: number;
}

interface Cell {
  run: string;
  run_id: string;
  task: string;
  threads: number;
  system: string;
  engine: string;
  model: string;
  task_type: string;
  n_rows: number;
  n_resamples: number;
  category: string | null;
  disturbed: string | null;
  stats: Record<string, Stat>;
}

interface Bench {
  source: string;
  runs: Record<string, string>;
  cells: Cell[];
}

const SYSTEMS: Array<[string, string, string]> = [
  ["r", "R tidymodels", "var(--trm-r, #4c6a8a)"],
  ["rust", "Rust", "var(--trm-rust, #b7410e)"],
  ["python", "Python wrapper", "var(--trm-python, #2f7f6f)"],
];

const PHASE_LABEL: Record<string, string> = {
  wf_fit_s: "workflow fit, cold (s)",
  warm_wf_fit_s: "workflow fit, warm (s)",
  cv_s: "10-fold cross-validation (s)",
  tune_s: "tuning grid (s)",
  wfs_s: "workflow set (s)",
  wall_s: "process wall time (s)",
  peak_wset_mb: "peak working set (MB)",
};

let cache: Promise<Bench> | null = null;

function load(): Promise<Bench> {
  if (!cache) {
    const root = (window as unknown as { __siteRoot?: string }).__siteRoot ?? "";
    cache = fetch(`${root}data/bench.json`).then((r) => {
      if (!r.ok) throw new Error(`bench.json: HTTP ${r.status}`);
      return r.json() as Promise<Bench>;
    });
  }
  return cache;
}

function svgEl<K extends keyof SVGElementTagNameMap>(name: K, attrs: Record<string, string | number> = {}): SVGElementTagNameMap[K] {
  const el = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
}

function fmt(x: number): string {
  if (x >= 100) return x.toFixed(0);
  if (x >= 10) return x.toFixed(1);
  if (x >= 1) return x.toFixed(2);
  if (x >= 0.01) return x.toFixed(3);
  return x.toExponential(1);
}

export function select(bench: Bench, el: HTMLElement): { phase: string; groups: Map<string, Cell[]> } {
  const phase = el.dataset.phase ?? "cv_s";
  const runs = el.dataset.runs ? new Set(el.dataset.runs.split(",").map((s) => s.trim())) : null;
  const threads = Number(el.dataset.threads ?? "1");
  const tasks = el.dataset.tasks ? new Set(el.dataset.tasks.split(",").map((s) => s.trim())) : null;
  const models = el.dataset.models ? new Set(el.dataset.models.split(",").map((s) => s.trim())) : null;
  const groups = new Map<string, Cell[]>();
  for (const c of bench.cells) {
    if (c.threads !== threads || !c.stats[phase]) continue;
    if (runs && !runs.has(c.run)) continue;
    if (tasks && !tasks.has(c.task)) continue;
    if (models && !models.has(c.model)) continue;
    const g = groups.get(c.task) ?? [];
    g.push(c);
    groups.set(c.task, g);
  }
  // order by size, then name
  const ordered = new Map([...groups.entries()].sort((a, b) => a[1][0].n_rows - b[1][0].n_rows || a[0].localeCompare(b[0])));
  return { phase, groups: ordered };
}

export function draw(bench: Bench, el: HTMLElement): void {
  const { phase, groups } = select(bench, el);
  el.textContent = "";
  if (groups.size === 0) {
    el.textContent = "No measured cells match this chart's selection.";
    return;
  }
  const rowH = 18;
  const groupPad = 14;
  const left = 230;
  const right = 110;
  const width = 860;
  const top = 28;
  const groupH = SYSTEMS.length * rowH + groupPad;
  const height = top + groups.size * groupH + 36;
  let min = Infinity;
  let max = -Infinity;
  for (const cells of groups.values()) {
    for (const c of cells) {
      const s = c.stats[phase];
      min = Math.min(min, s.q1 > 0 ? s.q1 : s.median);
      max = Math.max(max, s.q3);
    }
  }
  const lo = Math.floor(Math.log10(min));
  const hi = Math.ceil(Math.log10(max));
  const x = (v: number): number => left + ((Math.log10(v) - lo) / (hi - lo || 1)) * (width - left - right);

  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", class: "bench-chart" });
  const title = svgEl("title");
  title.textContent = `${el.dataset.title ?? ""} ${PHASE_LABEL[phase] ?? phase}, medians over cold trials with interquartile range`.trim();
  svg.appendChild(title);

  // axis grid at powers of ten
  for (let p = lo; p <= hi; p++) {
    const gx = x(10 ** p);
    svg.appendChild(svgEl("line", { x1: gx, x2: gx, y1: top - 6, y2: height - 30, class: "grid" }));
    const t = svgEl("text", { x: gx, y: height - 14, class: "axis", "text-anchor": "middle" });
    t.textContent = fmt(10 ** p);
    svg.appendChild(t);
  }
  const axisLabel = svgEl("text", { x: left + (width - left - right) / 2, y: height - 2, class: "axis-title", "text-anchor": "middle" });
  axisLabel.textContent = `${PHASE_LABEL[phase] ?? phase}, log scale`;
  svg.appendChild(axisLabel);

  const disturbedNotes: string[] = [];
  let gi = 0;
  for (const [task, cells] of groups) {
    const y0 = top + gi * groupH;
    const first = cells[0];
    const label = svgEl("text", { x: left - 8, y: y0 + rowH, class: "task", "text-anchor": "end" });
    label.textContent = `${task}`;
    svg.appendChild(label);
    const sub = svgEl("text", { x: left - 8, y: y0 + rowH * 2 - 2, class: "task-sub", "text-anchor": "end" });
    sub.textContent = `${first.n_rows.toLocaleString()} rows · ${first.model} · run ${first.run}`;
    svg.appendChild(sub);
    const byPhaseMedian: Record<string, number> = {};
    SYSTEMS.forEach(([sys, name, color], si) => {
      const c = cells.find((k) => k.system === sys);
      const y = y0 + si * rowH + 4;
      if (!c) return;
      const s = c.stats[phase];
      byPhaseMedian[sys] = s.median;
      svg.appendChild(svgEl("rect", { x: x(Math.min(s.median, 10 ** lo)), y, width: Math.max(1, x(s.median) - x(10 ** lo)), height: rowH - 6, fill: color, rx: 2 }));
      if (s.q1 > 0 && s.q3 > s.q1) {
        svg.appendChild(svgEl("line", { x1: x(s.q1), x2: x(s.q3), y1: y + (rowH - 6) / 2, y2: y + (rowH - 6) / 2, class: "iqr" }));
      }
      const v = svgEl("text", { x: x(s.median) + 5, y: y + rowH - 8, class: "value" });
      v.textContent = `${fmt(s.median)} (n=${s.n})${c.disturbed ? " †" : ""}`;
      svg.appendChild(v);
      const sysLabel = svgEl("text", { x: left + 4, y: y + rowH - 8, class: "system" });
      sysLabel.textContent = name;
      svg.appendChild(sysLabel);
      if (c.disturbed) disturbedNotes.push(`${task} (${name}): ${c.disturbed}`);
    });
    if (byPhaseMedian.r && byPhaseMedian.rust) {
      const ratio = byPhaseMedian.r / byPhaseMedian.rust;
      const r = svgEl("text", { x: width - right + 8, y: y0 + rowH + 4, class: "ratio" });
      r.textContent = `R / Rust ${ratio >= 10 ? ratio.toFixed(0) : ratio.toFixed(2)}×`;
      svg.appendChild(r);
      if (first.category) {
        const cat = svgEl("text", { x: width - right + 8, y: y0 + rowH * 2 + 2, class: "task-sub" });
        cat.textContent = first.category.split(" (")[0];
        svg.appendChild(cat);
      }
    }
    gi += 1;
  }
  el.appendChild(svg);
  const legend = document.createElement("p");
  legend.className = "bench-legend";
  legend.textContent =
    "Bars: median over cold trials; whisker: interquartile range; n: trials. Ratios are computed from the two medians shown. " +
    "Source: benchmarks/results/<run>/results.csv (runs " +
    [...new Set([...groups.values()].flat().map((c) => c.run_id))].join(", ") +
    ").";
  el.appendChild(legend);
  if (disturbedNotes.length) {
    const d = document.createElement("p");
    d.className = "bench-disturbed";
    d.textContent = "† " + [...new Set(disturbedNotes)].join("; ");
    el.appendChild(d);
  }
}

export function init(): void {
  const charts = Array.from(document.querySelectorAll<HTMLElement>("[data-bench-chart]"));
  if (charts.length === 0) return;
  load()
    .then((bench) => charts.forEach((el) => draw(bench, el)))
    .catch((e: unknown) => {
      for (const el of charts) el.textContent = `Chart data could not be loaded (${String(e)}); the numbers are in BENCHMARKS.md.`;
    });
}

if (typeof document !== "undefined") {
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
}
