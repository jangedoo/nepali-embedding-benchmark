import React, { useEffect, useMemo, useState } from "react";
import { Badge } from "./Badge";
import { coverage, formatCount, initialQuery, modelKey, rank, resultKey, visibleModels } from "../lib/catalog";
import type { Catalog, Model, Result, Task } from "../lib/types";

type Mode = "tasks" | "models" | "compare";
const fmt = (value: number) => value.toFixed(4);
const shortSha = (value: string) => value.slice(0, 8);
const revisionUrl = (repository: string, revision: string) => `https://huggingface.co/${repository}/tree/${revision}`;
const dataBase = typeof __NEB_BASE__ === "undefined" ? "/" : __NEB_BASE__;

export default function CatalogExplorer({ mode, compact = false }: { mode: Mode; compact?: boolean }) {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [error, setError] = useState("");
  const [showOlder, setShowOlder] = useState(false);

  useEffect(() => {
    fetch(`${dataBase}data/v3/catalog.json`)
      .then((response) => {
        if (!response.ok) throw new Error(`catalog request failed (${response.status})`);
        return response.json();
      })
      .then((value) => setCatalog(value as Catalog))
      .catch((reason: Error) => setError(reason.message));
  }, []);

  if (error) return <p role="alert">Could not load benchmark data: {error}</p>;
  if (!catalog) return <p aria-live="polite">Loading benchmark data…</p>;

  const historyControl = <label className="history-toggle"><input type="checkbox" checked={showOlder} onChange={(event) => setShowOlder(event.target.checked)} /> Show older revisions</label>;
  if (mode === "models") return <ModelCatalog catalog={catalog} showOlder={showOlder} historyControl={historyControl} />;
  if (mode === "compare") return <Comparison catalog={catalog} showOlder={showOlder} historyControl={historyControl} compact={compact} />;
  return <TaskCatalog catalog={catalog} showOlder={showOlder} historyControl={historyControl} compact={compact} />;
}

function TaskCatalog({ catalog, showOlder, historyControl, compact }: { catalog: Catalog; showOlder: boolean; historyControl: React.ReactNode; compact: boolean }) {
  const requestedTask = initialQuery("task");
  const [taskName, setTaskName] = useState(requestedTask);
  const activeTask = catalog.tasks.find((task) => task.name === taskName) || catalog.tasks[0];
  const requestedSubsets = initialQuery("subset").split(",").filter(Boolean);
  const [subsets, setSubsets] = useState(() => requestedSubsets.length ? requestedSubsets : activeTask?.subsets.map((item) => item.name) || []);
  const [split, setSplit] = useState(activeTask?.splits[0] || "test");
  const [expandedMetrics, setExpandedMetrics] = useState(false);
  const subsetPickerRef = React.useRef<HTMLDetailsElement>(null);

  useEffect(() => {
    function closeSubsetPicker(event: MouseEvent) {
      const picker = subsetPickerRef.current;
      if (picker?.open && !picker.contains(event.target as Node)) picker.open = false;
    }

    document.addEventListener("click", closeSubsetPicker);
    return () => document.removeEventListener("click", closeSubsetPicker);
  }, []);

  function selectTask(name: string) {
    const task = catalog.tasks.find((item) => item.name === name)!;
    setTaskName(name);
    setSubsets(task.subsets.map((item) => item.name));
    setSplit(task.splits[0] || "test");
  }

  function toggleSubset(name: string) {
    setSubsets(subsets.includes(name)
      ? subsets.filter((item) => item !== name)
      : activeTask.subsets.filter((item) => subsets.includes(item.name) || item.name === name).map((item) => item.name));
  }

  if (!activeTask) return <p className="empty">No benchmark tasks are available.</p>;
  const available = new Set(visibleModels(catalog.models, showOlder).map(modelKey));
  const selectedSubsets = activeTask.subsets.filter((item) => subsets.includes(item.name));
  const rows = catalog.results.filter((result) => result.task_name === activeTask.name && result.split === split && available.has(resultKey(result)));
  const modelCount = new Set(rows.map(resultKey)).size;
  const populatedSubsetCount = new Set(rows.map((result) => result.subset)).size;
  const hasSecondaryMetrics = rows.some((result) => Object.keys(result.metrics).some((metric) => metric !== activeTask.main_score));

  return <>
    {!compact && <section className="hero"><h1>Nepali embedding task results</h1><p>Browse every subset together. Each ranking remains local to its subset, split, and native main score.</p></section>}
    <section className="panel controls" aria-label="Task result controls">
      <label>Task<select value={activeTask.name} onChange={(event) => selectTask(event.target.value)}>{catalog.tasks.map((task) => <option key={task.name} value={task.name}>{task.display_name}</option>)}</select></label>
      <label>Split<select value={split} onChange={(event) => setSplit(event.target.value)}>{activeTask.splits.map((item) => <option key={item}>{item}</option>)}</select></label>
      <div className="view-picker">
        <span className="control-label">Subsets</span>
        <details ref={subsetPickerRef}>
          <summary>{subsets.length === activeTask.subsets.length ? `All ${activeTask.subsets.length} subsets` : `${selectedSubsets.length} of ${activeTask.subsets.length} subsets`}</summary>
          <fieldset>
            <legend className="sr-only">Choose task subsets</legend>
            <div className="view-picker-actions">
              <button type="button" onClick={() => setSubsets(activeTask.subsets.map((item) => item.name))}>Select all</button>
              <button type="button" onClick={() => setSubsets([])}>Clear</button>
            </div>
            <div className="view-options">{activeTask.subsets.map((item) => <label key={item.name}><input type="checkbox" checked={subsets.includes(item.name)} onChange={() => toggleSubset(item.name)} /> {item.name}</label>)}</div>
          </fieldset>
        </details>
      </div>
      {historyControl}
    </section>
    <section className="panel task-results">
      <div className="section-head"><div><h2>{activeTask.display_name}</h2><p>{activeTask.description}</p></div><a href={activeTask.dataset.url}>dataset {shortSha(activeTask.dataset.revision)}</a></div>
      <div className="task-summary" aria-label="Task summary">
        <span><strong>{populatedSubsetCount}</strong> / {activeTask.subsets.length} subsets with results</span>
        <span><strong>{modelCount}</strong> model {modelCount === 1 ? "revision" : "revisions"}</span>
        <span><strong>{activeTask.main_score}</strong> main score</span>
        {hasSecondaryMetrics && <button type="button" className="metrics-toggle" aria-expanded={expandedMetrics} onClick={() => setExpandedMetrics(!expandedMetrics)}>{expandedMetrics ? "Main score only" : "Show all metrics"}</button>}
      </div>
      {selectedSubsets.length ? <div className="subset-grid">{selectedSubsets.map((item) => {
        const subsetRows = rank(rows.filter((result) => result.subset === item.name));
        const headingId = `subset-${activeTask.name}-${item.name}`;
        return <section className="view-ranking" aria-labelledby={headingId} key={item.name}>
          <div className="subset-heading"><div><h3 id={headingId}>{item.name}</h3><p>{item.languages.join(" · ")}</p></div><span>{split} · {subsetRows.length} {subsetRows.length === 1 ? "result" : "results"}</span></div>
          {subsetRows.length ? <Ranking task={activeTask} results={subsetRows} models={catalog.models} expanded={expandedMetrics} /> : <p className="empty">No published result for this subset and split yet.</p>}
        </section>;
      })}</div> : <p className="empty">Select at least one subset.</p>}
    </section>
  </>;
}

function Ranking({ task, results, models, expanded }: { task: Task; results: Result[]; models: Model[]; expanded: boolean }) {
  const byKey = new Map(models.map((model) => [modelKey(model), model]));
  const metrics = [...new Set(results.flatMap((result) => Object.keys(result.metrics)))];
  const visibleMetrics = expanded ? [task.main_score, ...metrics.filter((metric) => metric !== task.main_score)] : [task.main_score];
  const bestScores = Object.fromEntries(visibleMetrics.map((metric) => [metric, Math.max(...results.map((result) => result.metrics[metric]).filter((score) => score !== undefined))]));
  return <div className="table-wrap"><table className={`ranking-table${expanded ? " expanded-metrics" : ""}`}><thead><tr><th scope="col"><span className="sr-only">Rank</span>#</th><th scope="col">Model</th>{visibleMetrics.map((metric) => <th scope="col" className={metric === task.main_score ? "main-metric" : undefined} key={metric}>{metric}{metric === task.main_score && <span className="sr-only"> (main score)</span>}</th>)}</tr></thead>
    <tbody>{results.map((result, index) => <RankingRow key={resultKey(result)} result={result} rank={index + 1} model={byKey.get(resultKey(result))} metrics={visibleMetrics} mainScore={task.main_score} bestScores={bestScores} />)}</tbody>
  </table></div>;
}

function RankingRow({ result, rank: position, model, metrics, mainScore, bestScores }: { result: Result; rank: number; model?: Model; metrics: string[]; mainScore: string; bestScores: Record<string, number> }) {
  const [open, setOpen] = useState(false);
  const panelId = `evidence-${React.useId().replaceAll(":", "")}`;
  return <>
    <tr>
      <td className="rank-cell">{position}</td>
      <th scope="row"><div className="model-cell"><a href={revisionUrl(result.model_name, result.model_revision)}>{result.model_name}</a><div className="model-meta"><code>{shortSha(result.model_revision)}</code><span title="Number of model parameters">{model ? formatCount(model.n_parameters) : "unknown"} params</span>{model && !model.is_latest && <span>older revision</span>}<button type="button" className="evidence-toggle" aria-expanded={open} aria-controls={panelId} aria-label={`Result details for ${result.model_name} ${shortSha(result.model_revision)}`} onClick={() => setOpen(!open)}>{open ? "Hide details" : "Details"}</button></div></div></th>
      {metrics.map((metric) => {
        const score = result.metrics[metric];
        const best = score !== undefined && score === bestScores[metric];
        const classes = ["score", metric === mainScore ? "main-metric" : "", best ? "best-score" : ""].filter(Boolean).join(" ");
        return <td className={classes} key={metric}>{best && <span className="sr-only">Best score: </span>}{score === undefined ? "—" : fmt(score)}{metric === mainScore && <Badge status={result.status} />}</td>;
      })}
    </tr>
    {open && <tr className="evidence-row"><td colSpan={metrics.length + 2}><EvidencePanel id={panelId} result={result} /></td></tr>}
  </>;
}

function EvidencePanel({ id, result }: { id: string; result: Result }) {
  const prompts = Object.entries(result.effective_prompts);
  return <section id={id} className="evidence-panel" aria-label={`Result details for ${result.model_name}`}>
    <div className="evidence-panel-head"><strong>Result details</strong><span className={`status-label ${result.status}`}>{result.status === "verified" ? "Maintainer verified" : "Community · unverified"}</span></div>
    <dl>
      <div><dt>Model revision</dt><dd><a href={revisionUrl(result.model_name, result.model_revision)}><code>{result.model_revision}</code></a></dd></div>
      <div><dt>Dataset revision</dt><dd><a href={`https://huggingface.co/datasets/${result.dataset_name}/tree/${result.dataset_revision}`}><code>{result.dataset_revision}</code></a></dd></div>
      <div><dt>MTEB version</dt><dd>{result.mteb_version}</dd></div>
      <div><dt>Evaluated</dt><dd>{result.evaluated_at || "unknown"}</dd></div>
      <div><dt>Effective prompts</dt><dd>{prompts.length ? <span className="prompt-list">{prompts.map(([kind, prompt]) => <span key={kind}><strong>{kind}</strong> <code>{prompt}</code></span>)}</span> : "none"}</dd></div>
      <div><dt>Result SHA-256</dt><dd><code>{result.result_sha256}</code></dd></div>
    </dl>
  </section>;
}

function ModelCatalog({ catalog, showOlder, historyControl }: { catalog: Catalog; showOlder: boolean; historyControl: React.ReactNode }) {
  const [search, setSearch] = useState("");
  const models = visibleModels(catalog.models, showOlder).filter((model) => model.repository.toLowerCase().includes(search.toLowerCase()));
  const groups = models.reduce((grouped, model) => {
    const revisions = grouped.get(model.repository) || [];
    revisions.push(model);
    grouped.set(model.repository, revisions);
    return grouped;
  }, new Map<string, Model[]>());
  return <><section className="hero"><p className="eyebrow">Published native MTEB evidence</p><h1>Models</h1></section><section className="panel controls"><label>Search models<input type="search" value={search} onChange={(event) => setSearch(event.target.value)} /></label>{historyControl}</section><section className="panel"><div className="card-grid">{[...groups].map(([repository, revisions]) => <article className="model-card" key={repository}><h2><a href={`https://huggingface.co/${repository}`}>{repository}</a></h2>{revisions.map((model) => { const count = coverage(catalog, model); return <section key={model.revision}><h3>{model.is_latest ? "Latest canonical revision" : "Historical revision"} <a href={revisionUrl(repository, model.revision)}><code>{shortSha(model.revision)}</code></a></h3><p>Parameters: {formatCount(model.n_parameters)}<br />Embedding dimension: {formatCount(model.embed_dim)}</p><p>Coverage: {count.complete} / {count.total} task views</p>{count.complete === 0 && <p className="empty">Awaiting results</p>}</section>; })}</article>)}</div>{models.length === 0 && <p className="empty">No published model revisions yet.</p>}</section></>;
}

function Comparison({ catalog, showOlder, historyControl, compact }: { catalog: Catalog; showOlder: boolean; historyControl: React.ReactNode; compact: boolean }) {
  const models = visibleModels(catalog.models, showOlder);
  const [selected, setSelected] = useState<string[]>([]);
  const [taskName, setTaskName] = useState(catalog.tasks[0]?.name || "");
  const task = catalog.tasks.find((item) => item.name === taskName);
  const rows = useMemo(() => catalog.results.filter((result) => result.task_name === taskName && selected.includes(`${result.model_name}@${result.model_revision}`)), [catalog, taskName, selected]);
  return <><section className={compact ? "" : "hero"}><h1>Task-local comparison</h1><p>Compare revisions without constructing an overall score.</p></section><section className="panel controls">{historyControl}<label>Task<select value={taskName} onChange={(event) => setTaskName(event.target.value)}>{catalog.tasks.map((item) => <option key={item.name} value={item.name}>{item.display_name}</option>)}</select></label></section><section className="panel"><fieldset><legend>Choose up to five model revisions</legend><div className="checks">{models.map((model) => { const key = modelKey(model); return <label key={key}><input type="checkbox" checked={selected.includes(key)} disabled={!selected.includes(key) && selected.length >= 5} onChange={() => setSelected(selected.includes(key) ? selected.filter((item) => item !== key) : [...selected, key])} /> {model.name} <code>{shortSha(model.revision)}</code></label>; })}</div></fieldset>{selected.length < 2 ? <p className="empty">Select at least two model revisions.</p> : task && task.subsets.map((subset) => task.splits.map((split) => { const view = rank(rows.filter((result) => result.subset === subset.name && result.split === split)); return <section key={`${subset.name}/${split}`}><h2>{subset.name} · {split} · {task.main_score}</h2>{view.length ? <ul>{view.map((result) => <li key={resultKey(result)}>{result.model_name} <code>{shortSha(result.model_revision)}</code>: {fmt(result.main_score)}</li>)}</ul> : <p className="empty">Missing results for selected revisions.</p>}</section>; }))}</section></>;
}
