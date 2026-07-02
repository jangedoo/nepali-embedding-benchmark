import React, { useEffect, useMemo, useState } from "react";
import { Badge } from "./Badge";
import { coverage, initialQuery, rank } from "../lib/catalog";
import type { Catalog, Model, Result, Task } from "../lib/types";

type Mode = "tasks" | "models" | "compare";

function fmt(value: number): string { return value.toFixed(4); }

export default function CatalogExplorer({ mode, compact = false }: { mode: Mode; compact?: boolean }) {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [taskId, setTaskId] = useState(() => initialQuery("task") || "");
  const [viewIds, setViewIds] = useState<string[] | null>(() => {
    const query = initialQuery("view");
    return query ? query.split(",").filter(Boolean) : null;
  });
  const [selected, setSelected] = useState<string[]>(() => (initialQuery("models") || "").split(",").filter(Boolean).slice(0, 5));

  useEffect(() => {
    fetch(`${__NEB_BASE__}data/v1/catalog.json`)
      .then((response) => {
        if (!response.ok) throw new Error(`catalog request failed (${response.status})`);
        return response.json();
      })
      .then((value) => setCatalog(value as Catalog))
      .catch((reason: Error) => setError(reason.message));
  }, []);

  const activeTask = catalog?.tasks.find((task) => task.id === taskId) || catalog?.tasks[0];
  const validViewIds = viewIds?.filter((id) => activeTask?.views.some((view) => view.id === id));
  const selectedViewIds = activeTask && (viewIds === null || (viewIds.length > 0 && validViewIds?.length === 0))
    ? activeTask.views.map((view) => view.id)
    : validViewIds || [];
  const visibleModels = useMemo(() => catalog?.models.filter((model) => `${model.display_name} ${model.hf_id}`.toLowerCase().includes(search.toLowerCase())) || [], [catalog, search]);

  useEffect(() => {
    if (activeTask && taskId !== activeTask.id) setTaskId(activeTask.id);
  }, [activeTask, taskId]);

  if (error) return <p role="alert">Could not load benchmark data: {error}</p>;
  if (!catalog) return <p aria-live="polite">Loading benchmark data…</p>;

  if (mode === "models") return <ModelCatalog catalog={catalog} models={visibleModels} search={search} setSearch={setSearch} />;
  if (mode === "compare") return <Comparison catalog={catalog} selected={selected} setSelected={setSelected} compact={compact} />;
  return <TaskCatalog catalog={catalog} activeTask={activeTask!} taskId={taskId} selectedViewIds={selectedViewIds} setTaskId={setTaskId} setViewIds={setViewIds} compact={compact} />;
}

function TaskCatalog({ catalog, activeTask, taskId, selectedViewIds, setTaskId, setViewIds, compact }: {
  catalog: Catalog; activeTask: Task; taskId: string; selectedViewIds: string[];
  setTaskId: (value: string) => void; setViewIds: (value: string[]) => void; compact: boolean;
}) {
  const selectedViews = activeTask.views.filter((view) => selectedViewIds.includes(view.id));

  function selectTask(id: string) {
    const task = catalog.tasks.find((candidate) => candidate.id === id)!;
    setTaskId(id);
    setViewIds(task.views.map((view) => view.id));
  }

  function toggleView(id: string) {
    setViewIds(selectedViewIds.includes(id)
      ? selectedViewIds.filter((selectedId) => selectedId !== id)
      : activeTask.views.filter((view) => selectedViewIds.includes(view.id) || view.id === id).map((view) => view.id));
  }

  return <>
    {!compact && <section className="hero"><h1>Nepali embedding task rankings</h1><p>Choose a task and compare model performance across its views.</p></section>}
    <section className="panel controls" aria-label="Task ranking controls">
      <label>Task<select value={taskId} onChange={(event) => selectTask(event.target.value)}>{catalog.tasks.map((task) => <option key={task.id} value={task.id}>{task.display_name}</option>)}</select></label>
      <div className="view-picker">
        <span className="control-label">Views</span>
        <details>
          <summary>{selectedViewIds.length === activeTask.views.length ? "All views" : `${selectedViewIds.length} of ${activeTask.views.length} views`}</summary>
          <fieldset>
            <legend className="sr-only">Choose task views</legend>
            <div className="view-picker-actions">
              <button type="button" onClick={() => setViewIds(activeTask.views.map((view) => view.id))}>Select all</button>
              <button type="button" onClick={() => setViewIds([])}>Clear</button>
            </div>
            <div className="view-options">{activeTask.views.map((view) => <label key={view.id}><input type="checkbox" checked={selectedViewIds.includes(view.id)} onChange={() => toggleView(view.id)} /> {view.id}</label>)}</div>
          </fieldset>
        </details>
      </div>
    </section>
    <section className="panel"><div className="section-head"><div><h2>{activeTask.display_name}</h2><p>{activeTask.description}</p></div><a href={activeTask.dataset.url}>dataset {activeTask.dataset.revision.slice(0, 8)}</a></div>
      {selectedViews.length ? selectedViews.map((view) => {
        const scores = rank(catalog.results.filter((result) => result.task_id === activeTask.id && result.view_id === view.id));
        const headingId = `view-${activeTask.id}-${view.id}`;
        return <section className="view-ranking" aria-labelledby={headingId} key={view.id}><h3 id={headingId}>{view.id}</h3>{scores.length ? <Ranking results={scores} models={catalog.models} /> : <p className="empty">No submitted results for this view yet.</p>}</section>;
      }) : <p className="empty">Select at least one view.</p>}
    </section>
  </>;
}

function Ranking({ results, models }: { results: Result[]; models: Model[] }) {
  const names = new Map(models.map((model) => [model.id, model.display_name]));
  return <div className="table-wrap"><table><thead><tr><th scope="col">Rank</th><th scope="col">Model</th><th scope="col">Metric</th><th scope="col">Score</th></tr></thead><tbody>{results.map((result, index) => <tr key={result.model_id}><td>{index + 1}</td><th scope="row">{names.get(result.model_id) || result.model_id}</th><td>{result.metric}</td><td className="score">{fmt(result.score)} <Badge status={result.status} /></td></tr>)}</tbody></table></div>;
}

function ModelCatalog({ catalog, models, search, setSearch }: { catalog: Catalog; models: Model[]; search: string; setSearch: (value: string) => void }) {
  return <><section className="hero"><p className="eyebrow">Pinned model catalog</p><h1>Models</h1></section><section className="panel"><label>Search models<input type="search" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Name or Hugging Face id" /></label><div className="card-grid">{models.map((model) => { const count = coverage(catalog, model.id); return <article className="model-card" key={model.id}><h2>{model.display_name}</h2><a href={model.homepage}>{model.hf_id}</a><p><code>{model.revision.slice(0, 12)}</code></p><p>Coverage: {count.complete} / {count.total} task views</p>{count.complete === 0 && <p className="empty">Awaiting results</p>}</article>; })}</div></section></>;
}

function Comparison({ catalog, selected, setSelected, compact }: { catalog: Catalog; selected: string[]; setSelected: (value: string[]) => void; compact: boolean }) {
  function toggle(id: string) { setSelected(selected.includes(id) ? selected.filter((value) => value !== id) : selected.length < 5 ? [...selected, id] : selected); }
  const rows = catalog.results.filter((result) => selected.includes(result.model_id));
  return <><section className={compact ? "" : "hero"}><p className="eyebrow">Two to five models</p><h1>Task-by-task comparison</h1></section><section className="panel"><fieldset><legend>Choose models ({selected.length}/5)</legend><div className="checks">{catalog.models.map((model) => <label key={model.id}><input type="checkbox" checked={selected.includes(model.id)} disabled={!selected.includes(model.id) && selected.length >= 5} onChange={() => toggle(model.id)} /> {model.display_name}</label>)}</div></fieldset>{selected.length < 2 ? <p className="empty">Select at least two models.</p> : <ComparisonTable catalog={catalog} rows={rows} selected={selected} />}</section></>;
}

function ComparisonTable({ catalog, rows, selected }: { catalog: Catalog; rows: Result[]; selected: string[] }) {
  const names = new Map(catalog.models.map((model) => [model.id, model.display_name]));
  const keys = [...new Set(rows.map((row) => `${row.task_id}/${row.view_id}`))].sort();
  return <div className="table-wrap"><table className="comparison-table"><thead><tr><th>Task / view</th>{selected.map((id) => <th key={id}>{names.get(id)}</th>)}</tr></thead><tbody>{keys.map((key) => {
    const results = rows.filter((row) => `${row.task_id}/${row.view_id}` === key);
    const bestScore = Math.max(...results.map((result) => result.score));
    const metric = results[0].metric;
    return <tr key={key}><th scope="row"><span>{key}</span><span className="metric-name">Metric: {metric}</span></th>{selected.map((id) => {
      const result = results.find((row) => row.model_id === id);
      const isBest = result?.score === bestScore;
      return <td className={isBest ? "best-result" : undefined} key={id}>{result ? <><span className="sr-only">{isBest ? "Best score: " : "Score: "}</span>{fmt(result.score)} {result.status === "community" && <Badge status="community" />}</> : <span className="missing">missing</span>}</td>;
    })}</tr>;
  })}</tbody></table></div>;
}
