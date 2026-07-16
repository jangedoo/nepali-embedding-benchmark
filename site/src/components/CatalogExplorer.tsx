import React, { useEffect, useMemo, useState } from "react";
import { Badge } from "./Badge";
import { coverage, formatCount, initialQuery, modelKey, rank, resultKey, visibleModels } from "../lib/catalog";
import type { Catalog, Model, Result, Task } from "../lib/types";

type Mode = "tasks" | "models" | "compare";
const fmt = (value: number) => value.toFixed(4);
const shortSha = (value: string) => value.slice(0, 8);
const revisionUrl = (repository: string, revision: string) => `https://huggingface.co/${repository}/tree/${revision}`;
const externalLinkProps = { target: "_blank", rel: "noreferrer" } as const;
const metricCollator = new Intl.Collator("en", { numeric: true, sensitivity: "base" });
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
      <div className="section-head"><div><h2>{activeTask.display_name}</h2><p>{activeTask.description}</p></div><a href={activeTask.dataset.url} {...externalLinkProps}>dataset {shortSha(activeTask.dataset.revision)}</a></div>
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
  const metrics = [...new Set(results.flatMap((result) => Object.keys(result.metrics)))].sort(metricCollator.compare);
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
      <th scope="row"><div className="model-cell"><a href={revisionUrl(result.model_name, result.model_revision)} {...externalLinkProps}>{result.model_name}</a><div className="model-meta"><code>{shortSha(result.model_revision)}</code><span title="Number of model parameters">{model ? formatCount(model.n_parameters) : "unknown"} params</span>{model && !model.is_latest && <span>older revision</span>}<button type="button" className="evidence-toggle" aria-expanded={open} aria-controls={panelId} aria-label={`Result details for ${result.model_name} ${shortSha(result.model_revision)}`} onClick={() => setOpen(!open)}>{open ? "Hide details" : "Details"}</button></div></div></th>
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
      <div><dt>Model revision</dt><dd><a href={revisionUrl(result.model_name, result.model_revision)} {...externalLinkProps}><code>{result.model_revision}</code></a></dd></div>
      <div><dt>Dataset revision</dt><dd><a href={`https://huggingface.co/datasets/${result.dataset_name}/tree/${result.dataset_revision}`} {...externalLinkProps}><code>{result.dataset_revision}</code></a></dd></div>
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
  return <><section className="hero"><h1>Models</h1></section><section className="panel controls"><label>Search models<input type="search" value={search} onChange={(event) => setSearch(event.target.value)} /></label>{historyControl}</section><section className="panel"><div className="card-grid">{[...groups].map(([repository, revisions]) => <article className="model-card" key={repository}><h2><a href={`https://huggingface.co/${repository}`} {...externalLinkProps}>{repository}</a></h2>{revisions.map((model) => {
    const count = coverage(catalog, model);
    return <section className="model-revision" key={model.revision}>
      <div className="model-revision-line"><a href={revisionUrl(repository, model.revision)} aria-label={`${repository} revision ${shortSha(model.revision)}`} {...externalLinkProps}><code>{shortSha(model.revision)}</code></a>{!model.is_latest && <span>Historical revision</span>}</div>
      <dl className="model-stats">
        <div><dt>Parameters</dt><dd>{formatCount(model.n_parameters)}</dd></div>
        <div><dt>Embedding dimension</dt><dd>{formatCount(model.embed_dim)}</dd></div>
        <div className="coverage-stat"><dt>Coverage</dt><dd>{count.complete} / {count.total} task views</dd></div>
      </dl>
      {count.complete === 0 && <p className="empty">Awaiting results</p>}
    </section>;
  })}</article>)}</div>{models.length === 0 && <p className="empty">No published model revisions yet.</p>}</section></>;
}

function Comparison({ catalog, showOlder, historyControl, compact }: { catalog: Catalog; showOlder: boolean; historyControl: React.ReactNode; compact: boolean }) {
  const validModelKeys = new Set(catalog.models.map(modelKey));
  const requestedTask = initialQuery("task");
  const initialTask = catalog.tasks.find((item) => item.name === requestedTask) || catalog.tasks[0];
  const requestedModels = [...new Set(initialQuery("models").split(",").filter((key) => validModelKeys.has(key)))].slice(0, 5);
  const requestedMetrics = initialQuery("metrics").split(",").filter(Boolean);
  const [selected, setSelected] = useState<string[]>(requestedModels);
  const [taskName, setTaskName] = useState(initialTask?.name || "");
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(() => {
    if (!initialTask) return [];
    const available = comparisonMetrics(catalog, initialTask);
    return [initialTask.main_score, ...available.filter((metric) => metric !== initialTask.main_score && requestedMetrics.includes(metric))];
  });
  const [search, setSearch] = useState("");
  const pickerRef = React.useRef<HTMLDetailsElement>(null);
  const task = catalog.tasks.find((item) => item.name === taskName);
  const availableMetrics = task ? comparisonMetrics(catalog, task) : [];
  const candidateModels = visibleModels(catalog.models, showOlder).filter((model) => `${model.name} ${model.revision}`.toLowerCase().includes(search.toLowerCase()));
  const selectedModels = selected.map((key) => catalog.models.find((model) => modelKey(model) === key)).filter((model): model is Model => Boolean(model));
  const rows = useMemo(() => catalog.results.filter((result) => result.task_name === taskName && selected.includes(resultKey(result))), [catalog, taskName, selected]);

  useEffect(() => {
    if (pickerRef.current) pickerRef.current.open = true;
  }, []);

  useEffect(() => {
    if (typeof window === "undefined" || !task) return;
    const params = new URLSearchParams(window.location.search);
    params.set("task", task.name);
    if (selected.length) params.set("models", selected.join(","));
    else params.delete("models");
    const secondaryMetrics = selectedMetrics.filter((metric) => metric !== task.main_score);
    if (secondaryMetrics.length) params.set("metrics", secondaryMetrics.join(","));
    else params.delete("metrics");
    const query = params.toString();
    const nextUrl = `${window.location.pathname}${query ? `?${query}` : ""}${window.location.hash}`;
    if (`${window.location.pathname}${window.location.search}${window.location.hash}` !== nextUrl) window.history.replaceState({}, "", nextUrl);
  }, [task, selected, selectedMetrics]);

  function selectTask(name: string) {
    const nextTask = catalog.tasks.find((item) => item.name === name);
    setTaskName(name);
    setSelectedMetrics(nextTask ? [nextTask.main_score] : []);
  }

  function toggleModel(key: string) {
    if (selected.includes(key)) {
      setSelected(selected.filter((item) => item !== key));
      return;
    }
    if (selected.length >= 5) return;
    setSelected([...selected, key]);
  }

  function toggleMetric(metric: string) {
    if (!task || metric === task.main_score) return;
    setSelectedMetrics(selectedMetrics.includes(metric)
      ? selectedMetrics.filter((item) => item !== metric)
      : availableMetrics.filter((item) => selectedMetrics.includes(item) || item === metric));
  }

  if (!task) return <p className="empty">No benchmark tasks are available.</p>;
  const viewCount = task.subsets.length * task.splits.length;

  return <div className="comparison-page">
    {!compact && <section className="hero comparison-hero"><p className="eyebrow">Side-by-side task results</p><h1>Compare models</h1><p>Choose a task, then compare up to five model revisions across every subset and split.</p></section>}
    <section className="panel comparison-panel" aria-label="Model comparison">
      <div className="comparison-toolbar">
        <label>Task<select value={taskName} onChange={(event) => selectTask(event.target.value)}>{catalog.tasks.map((item) => <option key={item.name} value={item.name}>{item.display_name}</option>)}</select></label>
        <div className="metric-select view-picker">
          <span className="control-label">Metrics</span>
          <details>
            <summary>{selectedMetrics.length === 1 ? "Main score only" : `${selectedMetrics.length} metrics`}</summary>
            <fieldset>
              <legend className="sr-only">Choose comparison metrics</legend>
              <div className="view-picker-actions"><button type="button" onClick={() => setSelectedMetrics([task.main_score])}>Main score only</button></div>
              <div className="view-options">{availableMetrics.map((metric) => <label key={metric}><input type="checkbox" checked={selectedMetrics.includes(metric)} disabled={metric === task.main_score} onChange={() => toggleMetric(metric)} /> {metric}{metric === task.main_score ? " (main)" : ""}</label>)}</div>
            </fieldset>
          </details>
        </div>
        {historyControl}
      </div>

      <details className="comparison-editor" ref={pickerRef}>
        <summary><span>Choose model revisions</span><span className="comparison-summary">{selected.length}/5 selected</span></summary>
        <div className="comparison-model-picker">
          <label className="model-search">Find a model<input type="search" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Repository or revision" /></label>
          <fieldset>
            <legend className="sr-only">Choose up to five model revisions</legend>
            <div className="model-choice-grid">{candidateModels.map((model) => {
              const key = modelKey(model);
              const complete = new Set(catalog.results.filter((result) => result.task_name === task.name && resultKey(result) === key).map((result) => `${result.subset}/${result.split}`)).size;
              return <label className={`model-choice${selected.includes(key) ? " selected" : ""}`} key={key}>
                <input type="checkbox" checked={selected.includes(key)} disabled={!selected.includes(key) && selected.length >= 5} onChange={() => toggleModel(key)} />
                <span><strong>{model.name}</strong><span className="model-choice-meta"><code>{shortSha(model.revision)}</code><span>{formatCount(model.n_parameters)} params</span><span>{complete}/{viewCount} results</span>{!model.is_latest && <span>historical</span>}</span></span>
              </label>;
            })}</div>
          </fieldset>
          {candidateModels.length === 0 && <p className="empty">No model revision matches “{search}”.</p>}
          {selected.length >= 5 && <p className="selection-limit" role="status">Five revisions selected. Remove one to choose another.</p>}
        </div>
      </details>

      {selectedModels.length > 0 && <div className="selected-models" aria-label="Selected model revisions">{selectedModels.map((model) => <span className="selected-model" key={modelKey(model)}><span><strong>{model.name}</strong> <code>{shortSha(model.revision)}</code>{!model.is_latest && <small>historical</small>}</span><button type="button" aria-label={`Remove ${model.name} ${shortSha(model.revision)}`} onClick={() => toggleModel(modelKey(model))}>×</button></span>)}</div>}

      {selected.length < 2
        ? <div className="comparison-empty"><strong>{selected.length === 0 ? "Choose at least two model revisions" : "Choose one more model revision"}</strong><p>The comparison will appear here with one row per subset and split.</p></div>
        : <ComparisonTable task={task} results={rows} models={selectedModels} metrics={selectedMetrics} />}
    </section>
  </div>;
}

function comparisonMetrics(catalog: Catalog, task: Task): string[] {
  const metrics = [...new Set(catalog.results.filter((result) => result.task_name === task.name).flatMap((result) => Object.keys(result.metrics)))].sort(metricCollator.compare);
  return [task.main_score, ...metrics.filter((metric) => metric !== task.main_score)];
}

function ComparisonTable({ task, results, models, metrics }: { task: Task; results: Result[]; models: Model[]; metrics: string[] }) {
  return <div className="comparison-table-wrap"><table className="comparison-table">
    <caption className="sr-only">{task.display_name} scores by subset, split, and model revision</caption>
    <thead><tr><th scope="col">Subset / split</th>{models.map((model) => <th scope="col" key={modelKey(model)}><div className="comparison-model-header"><a href={revisionUrl(model.repository, model.revision)} {...externalLinkProps}>{model.name}</a><code>{shortSha(model.revision)}</code><span>{formatCount(model.n_parameters)} params · {formatCount(model.embed_dim)} dimensions</span>{!model.is_latest && <span>Historical revision</span>}</div></th>)}</tr></thead>
    <tbody>{task.subsets.flatMap((subset) => task.splits.map((split) => <ComparisonRow key={`${subset.name}/${split}`} task={task} subset={subset} split={split} metrics={metrics} models={models} results={results.filter((result) => result.subset === subset.name && result.split === split)} />))}</tbody>
  </table></div>;
}

function ComparisonRow({ task, subset, split, metrics, models, results }: { task: Task; subset: Task["subsets"][number]; split: string; metrics: string[]; models: Model[]; results: Result[] }) {
  const [openResult, setOpenResult] = useState<string | null>(null);
  const detailsId = `comparison-evidence-${React.useId().replaceAll(":", "")}`;
  const bestScores = Object.fromEntries(metrics.map((metric) => {
    const scores = results.map((result) => result.metrics[metric]).filter((score) => score !== undefined);
    return [metric, scores.length >= 2 ? Math.max(...scores) : undefined];
  }));
  const detailedResult = results.find((result) => resultKey(result) === openResult);
  return <>
    <tr className="comparison-view-row">
      <th scope="row"><strong>{subset.name}</strong><span>{split}</span><small>{subset.languages.join(" · ")}</small></th>
      {models.map((model) => {
        const key = modelKey(model);
        const result = results.find((item) => resultKey(item) === key);
        if (!result) return <td className="missing-result" key={key}><span>Missing result</span></td>;
        const detailsOpen = openResult === key;
        return <td key={key}><div className="comparison-result">{metrics.map((metric) => {
          const score = result.metrics[metric];
          const primary = metric === task.main_score;
          const best = score !== undefined && score === bestScores[metric];
          return <div className={`comparison-metric${primary ? " primary-metric" : " secondary-metric"}${best ? " metric-winner" : ""}`} key={metric} data-metric={metric}>
            <span className="comparison-metric-name">{metric}{primary && <span className="sr-only"> (main score)</span>}</span>
            <strong>{best && <span className="sr-only">Best score: </span>}{score === undefined ? "—" : fmt(score)}</strong>
          </div>;
        })}<div className="comparison-result-meta"><Badge status={result.status} /><button type="button" className="evidence-toggle" aria-expanded={detailsOpen} aria-controls={detailsId} aria-label={`Result details for ${result.model_name} ${shortSha(result.model_revision)} on ${subset.name} ${split}`} onClick={() => setOpenResult(detailsOpen ? null : key)}>{detailsOpen ? "Hide details" : "Details"}</button></div></div></td>;
      })}
    </tr>
    {detailedResult && <tr className="evidence-row comparison-evidence-row"><td colSpan={models.length + 1}><EvidencePanel id={detailsId} result={detailedResult} /></td></tr>}
  </>;
}
