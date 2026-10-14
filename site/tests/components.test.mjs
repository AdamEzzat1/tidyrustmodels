// Smoke tests for the compiled bundles (assets/*.js) against a jsdom page.
// Run after `npm run build`: `npm test` (node --test).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { JSDOM } from "jsdom";

function page(html, { bench } = {}) {
  // a URL gives the page an origin, so jsdom provides localStorage as a browser would
  const dom = new JSDOM(html, { runScripts: "outside-only", pretendToBeVisual: true, url: "http://localhost/" });
  const { window } = dom;
  window.__siteRoot = "";
  if (bench) {
    window.fetch = async () => ({ ok: true, json: async () => bench });
  }
  return dom;
}

async function run(dom, bundle) {
  const code = readFileSync(new URL(`../assets/${bundle}.js`, import.meta.url), "utf8");
  // the bundles are ESM with no imports; evaluate them as classic scripts
  dom.window.eval(code.replace(/export\s*\{[^}]*\};?/g, ""));
  // the components initialise on DOMContentLoaded when the document is still loading
  if (dom.window.document.readyState === "loading") {
    await new Promise((resolve) => dom.window.addEventListener("DOMContentLoaded", resolve));
  }
  await new Promise((resolve) => setTimeout(resolve, 10));
}

test("table-filter hides non-matching rows and reports the count", async () => {
  const dom = page(`<div class="filterable"><table><thead><tr><th>a</th></tr></thead>
    <tbody><tr><td>rsample vfold_cv</td></tr><tr><td>yardstick rmse</td></tr></tbody></table></div>`);
  await run(dom, "table-filter");
  const input = dom.window.document.querySelector(".table-filter input");
  assert.ok(input, "filter box added");
  input.value = "rmse";
  input.dispatchEvent(new dom.window.Event("input"));
  const rows = [...dom.window.document.querySelectorAll("tbody tr")];
  assert.deepEqual(rows.map((r) => r.hidden), [true, false]);
  assert.equal(dom.window.document.querySelector(".table-filter-count").textContent, "1 of 2 rows");
});

test("code-tabs stores the chosen tab and survives a storage failure", async () => {
  const dom = page(`<div class="panel-tabset" data-group="language"><ul class="nav nav-tabs" role="tablist">
    <li class="nav-item"><a class="nav-link active" role="tab">Rust</a></li>
    <li class="nav-item"><a class="nav-link" role="tab">Python</a></li></ul></div>`);
  await run(dom, "code-tabs");
  const links = [...dom.window.document.querySelectorAll("a.nav-link")];
  links[1].click();
  assert.equal(dom.window.localStorage.getItem("trm.language"), "Python");
  // storage that throws must not break the page
  Object.defineProperty(dom.window, "localStorage", { get() { throw new Error("blocked"); } });
  assert.doesNotThrow(() => links[0].click());
});

test("bench-chart draws one bar group per task with the R / Rust ratio", async () => {
  const bench = {
    source: "test",
    runs: { "run-x": "X" },
    cells: [
      { run: "X", run_id: "run-x", task: "t1", threads: 1, system: "r", engine: "lm", model: "linear_reg", task_type: "regression", n_rows: 100, n_resamples: 10, category: "framework-overhead (lm vs port)", disturbed: null, stats: { cv_s: { median: 4, q1: 3.5, q3: 4.5, n: 10 } } },
      { run: "X", run_id: "run-x", task: "t1", threads: 1, system: "rust", engine: "native", model: "linear_reg", task_type: "regression", n_rows: 100, n_resamples: 10, category: "framework-overhead (lm vs port)", disturbed: "trials 5–6 disturbed", stats: { cv_s: { median: 0.5, q1: 0.4, q3: 0.6, n: 10 } } },
    ],
  };
  const dom = page(`<div data-bench-chart data-phase="cv_s" data-threads="1"></div>`, { bench });
  await run(dom, "bench-chart");
  await new Promise((r) => setTimeout(r, 20));
  const el = dom.window.document.querySelector("[data-bench-chart]");
  assert.ok(el.querySelector("svg.bench-chart"), "svg drawn");
  assert.match(el.textContent, /R \/ Rust 8\.00×/);
  assert.match(el.querySelector(".bench-disturbed").textContent, /trials 5–6 disturbed/);
});
