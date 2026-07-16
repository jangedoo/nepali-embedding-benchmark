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
  const [subset, setSubset] = useState(activeTask?.subsets[0]?.name || "");
  const [split, setSplit] = useState(activeTask?.splits[0] || "test");

  function selectTask(name: string) {
    const task = catalog.tasks.find((item) => item.name === name)!;
    setTaskName(name);
    setSubset(task.subsets[0]?.name || "");
    setSplit(task.splits[0] || "test");
  }

  if (!activeTask) return <p className="empty">No benchmark tasks are available.</p>;
  const available = new Set(visibleModels(catalog.models, showOlder).map(modelKey));
  const rows = catalog.results.filter((result) => result.task_name === activeTask.name && result.subset === subset && result.split === split && available.has(resultKey(result)));

  return <>
    {!compact && <section className="hero"><h1>Nepali embedding task results</h1><p>Rankings exist only inside the selected task, subset, split, and native main score.</p></section>}
    <section className="panel controls" aria-label="Task result controls">
      <label>Task<select value={activeTask.name} onChange={(event) => selectTask(event.target.value)}>{catalog.tasks.map((task) => <option key={task.name} value={task.name}>{task.display_name}</option>)}</select></label>
      <label>Subset<select value={subset} onChange={(event) => setSubset(event.target.value)}>{activeTask.subsets.map((item) => <option key={item.name}>{item.name}</option>)}</select></label>
      <label>Split<select value={split} onChange={(event) => setSplit(event.target.value)}>{activeTask.splits.map((item) => <option key={item}>{item}</option>)}</select></label>
      {historyControl}
    </section>
    <section className="panel">
      <div className="section-head"><div><h2>{activeTask.display_name}</h2><p>{activeTask.description}</p></div><a href={activeTask.dataset.url}>dataset {shortSha(activeTask.dataset.revision)}</a></div>
      <h3>{subset} · {split}</h3>
      {rows.length ? <Ranking task={activeTask} results={rank(rows)} models={catalog.models} /> : <p className="empty">No published result for this task view yet.</p>}
    </section>
  </>;
}

function Ranking({ task, results, models }: { task: Task; results: Result[]; models: Model[] }) {
  const [expanded, setExpanded] = useState(false);
  const byKey = new Map(models.map((model) => [modelKey(model), model]));
  const metrics = [...new Set(results.flatMap((result) => Object.keys(result.metrics)))];
  const visibleMetrics = expanded ? [task.main_score, ...metrics.filter((metric) => metric !== task.main_score)] : [task.main_score];
  return <>
    {metrics.length > 1 && <button type="button" aria-expanded={expanded} onClick={() => setExpanded(!expanded)}>{expanded ? "Show main score only" : "Show all native metrics"}</button>}
    <div className="table-wrap"><table><thead><tr><th scope="col">Rank</th><th scope="col">Model revision</th>{visibleMetrics.map((metric) => <th scope="col" key={metric}>{metric}{metric === task.main_score && <span className="sr-only"> (main score)</span>}</th>)}</tr></thead>
      <tbody>{results.map((result, index) => { const model = byKey.get(resultKey(result)); return <tr key={resultKey(result)}><td>{index + 1}</td><th scope="row"><a href={revisionUrl(result.model_name, result.model_revision)}>{result.model_name}</a> <code>{shortSha(result.model_revision)}</code> <EvidenceDetails result={result} /></th>{visibleMetrics.map((metric) => <td className="score" key={metric}>{result.metrics[metric] === undefined ? "—" : fmt(result.metrics[metric])}{metric === task.main_score && <Badge status={result.status} />}{model && !model.is_latest && <span className="sr-only"> historical revision</span>}</td>)}</tr>; })}</tbody>
    </table></div>
  </>;
}

function EvidenceDetails({ result }: { result: Result }) {
  return <details className="evidence-details"><summary aria-label={`Evidence details for ${result.model_name} ${shortSha(result.model_revision)}`}>details</summary><div className="model-tooltip">
    <span>Status: {result.status === "verified" ? "maintainer-verified" : "community-unverified"}</span>
    <span>Model SHA: <code>{result.model_revision}</code></span>
    <span>Dataset SHA: <a href={`https://huggingface.co/datasets/${result.dataset_name}/tree/${result.dataset_revision}`}><code>{result.dataset_revision}</code></a></span>
    <span>MTEB: {result.mteb_version}</span>
    <span>Prompts: {Object.keys(result.effective_prompts).length ? JSON.stringify(result.effective_prompts) : "none"}</span>
    <span>Result SHA-256: <code>{result.result_sha256}</code></span>
  </div></details>;
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
