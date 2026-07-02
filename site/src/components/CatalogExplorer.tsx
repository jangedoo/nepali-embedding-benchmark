import React, { useEffect, useMemo, useState } from "react";
import { Badge } from "./Badge";
import { coverage, formatCount, initialQuery, rank } from "../lib/catalog";
import type { Catalog, Model, Result, Task, View } from "../lib/types";

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
    fetch(`${__NEB_BASE__}data/v2/catalog.json`)
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
  const viewPickerRef = React.useRef<HTMLDetailsElement>(null);
  const selectedViews = activeTask.views.filter((view) => selectedViewIds.includes(view.id));

  useEffect(() => {
    function closeViewPicker(event: MouseEvent) {
      const picker = viewPickerRef.current;
      if (picker?.open && !picker.contains(event.target as Node)) picker.open = false;
    }

    document.addEventListener("click", closeViewPicker);
    return () => document.removeEventListener("click", closeViewPicker);
  }, []);

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
        <details ref={viewPickerRef}>
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
        const headingId = `view-${activeTask.id}-${view.id}`;
        const results = catalog.results.filter((result) => result.task_id === activeTask.id && result.view_id === view.id);
        return <section className="view-ranking" aria-labelledby={headingId} key={`${activeTask.id}/${view.id}`}><h3 id={headingId}>{view.id}</h3>{results.length ? <ViewResults results={results} models={catalog.models} view={view} /> : <p className="empty">No submitted results for this view yet.</p>}</section>;
      }) : <p className="empty">Select at least one view.</p>}
    </section>
  </>;
}

function ViewResults({ results, models, view }: { results: Result[]; models: Model[]; view: View }) {
  const [selectedMetrics, setSelectedMetrics] = useState([...view.metrics]);
  const secondary = view.metrics.filter((metric) => metric !== view.primary_metric);
  function toggleMetric(metric: string) {
    setSelectedMetrics(selectedMetrics.includes(metric)
      ? selectedMetrics.filter((selected) => selected !== metric)
      : view.metrics.filter((candidate) => selectedMetrics.includes(candidate) || candidate === metric));
  }
  return <>
    {secondary.length > 0 && <fieldset className="metric-picker"><legend>Table metrics</legend>{view.metrics.map((metric) => <label key={metric}><input type="checkbox" checked={selectedMetrics.includes(metric)} disabled={metric === view.primary_metric} onChange={() => toggleMetric(metric)} /> {metric}{metric === view.primary_metric ? " (primary)" : ""}</label>)}</fieldset>}
    <Ranking results={rank(results, view.primary_metric)} models={models} metrics={selectedMetrics} primaryMetric={view.primary_metric} />
  </>;
}

function Ranking({ results, models, metrics, primaryMetric }: { results: Result[]; models: Model[]; metrics: string[]; primaryMetric: string }) {
  const byId = new Map(models.map((model) => [model.id, model]));
  return <div className="table-wrap"><table><thead><tr><th scope="col">Rank</th><th scope="col">Model</th>{metrics.map((metric) => <th scope="col" key={metric}>{metric}{metric === primaryMetric && <span className="sr-only"> (primary metric)</span>}</th>)}</tr></thead><tbody>{results.map((result, index) => { const model = byId.get(result.model_id); return <tr key={result.model_id}><td>{index + 1}</td><th scope="row"><span>{model?.display_name || result.model_id}</span>{model && <ModelInfo model={model} />}</th>{metrics.map((metric) => <td className="score" key={metric}>{fmt(result.metrics[metric])} {metric === primaryMetric && result.status === "community" && <Badge status="community" />}</td>)}</tr>; })}</tbody></table></div>;
}

function ModelInfo({ model }: { model: Model }) {
  const popoverId = `model-info-${React.useId().replaceAll(":", "")}`;
  const triggerRef = React.useRef<HTMLButtonElement>(null);
  const popoverRef = React.useRef<HTMLDivElement>(null);

  useEffect(() => {
    const trigger = triggerRef.current;
    const popover = popoverRef.current;
    if (!trigger || !popover) return;
    function position() {
      const triggerBounds = trigger!.getBoundingClientRect();
      const width = popover!.offsetWidth || 220;
      const height = popover!.offsetHeight || 100;
      const left = Math.min(Math.max(8, triggerBounds.left), window.innerWidth - width - 8);
      const below = triggerBounds.bottom + 6;
      const top = below + height <= window.innerHeight - 8 ? below : Math.max(8, triggerBounds.top - height - 6);
      popover!.style.left = `${left}px`;
      popover!.style.top = `${top}px`;
    }
    function handleToggle(event: Event) {
      if ((event as Event & { newState?: string }).newState === "open") position();
    }
    popover.addEventListener("toggle", handleToggle);
    window.addEventListener("resize", position);
    return () => {
      popover.removeEventListener("toggle", handleToggle);
      window.removeEventListener("resize", position);
    };
  }, []);

  return <span className="model-info"><button ref={triggerRef} type="button" className="model-info-trigger" popoverTarget={popoverId} aria-label={`Model details for ${model.display_name}`} title="Model details">?</button><div ref={popoverRef} id={popoverId} className="model-tooltip" popover="auto"><span>Parameters: {formatCount(model.parameter_count)}</span><span>Vocabulary: {formatCount(model.vocab_size)}</span><a href={`https://huggingface.co/${model.hf_id}`}>{model.hf_id}</a></div></span>;
}

function ModelCatalog({ catalog, models, search, setSearch }: { catalog: Catalog; models: Model[]; search: string; setSearch: (value: string) => void }) {
  return <><section className="hero"><p className="eyebrow">Pinned model catalog</p><h1>Models</h1></section><section className="panel"><label>Search models<input type="search" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Name or Hugging Face id" /></label><div className="card-grid">{models.map((model) => { const count = coverage(catalog, model.id); return <article className="model-card" key={model.id}><h2>{model.display_name}</h2><a href={model.homepage}>{model.hf_id}</a><p><code>{model.revision.slice(0, 12)}</code></p><p>Parameters: {formatCount(model.parameter_count)}<br />Vocabulary: {formatCount(model.vocab_size)}</p><p>Coverage: {count.complete} / {count.total} task views</p>{count.complete === 0 && <p className="empty">Awaiting results</p>}</article>; })}</div></section></>;
}

function Comparison({ catalog, selected, setSelected, compact }: { catalog: Catalog; selected: string[]; setSelected: (value: string[]) => void; compact: boolean }) {
  const [selectedTaskIds, setSelectedTaskIds] = useState(() => catalog.tasks.map((task) => task.id));
  const [metricsByTask, setMetricsByTask] = useState<Record<string, string[]>>(() => Object.fromEntries(catalog.tasks.map((task) => [task.id, taskMetrics(task)])));
  function toggle(id: string) { setSelected(selected.includes(id) ? selected.filter((value) => value !== id) : selected.length < 5 ? [...selected, id] : selected); }
  function toggleTask(id: string) {
    setSelectedTaskIds((current) => current.includes(id)
      ? current.length > 1 ? current.filter((value) => value !== id) : current
      : catalog.tasks.filter((task) => current.includes(task.id) || task.id === id).map((task) => task.id));
  }
  function toggleMetric(taskId: string, metric: string) {
    const task = catalog.tasks.find((candidate) => candidate.id === taskId)!;
    if (primaryMetrics(task).includes(metric)) return;
    setMetricsByTask((current) => {
      const selectedMetrics = current[taskId];
      const next = selectedMetrics.includes(metric)
        ? selectedMetrics.filter((value) => value !== metric)
        : taskMetrics(task).filter((value) => selectedMetrics.includes(value) || value === metric);
      return { ...current, [taskId]: next };
    });
  }
  const rows = catalog.results.filter((result) => selected.includes(result.model_id));
  const selectedMetricCount = selectedTaskIds.reduce((count, id) => count + metricsByTask[id].length, 0);
  return <>
    <section className={compact ? "" : "hero"}><p className="eyebrow">Two to five models</p><h1>Task-by-task comparison</h1></section>
    <section className="panel comparison-panel">
      <div className="comparison-controls comparison-selection-controls">
        <fieldset><legend>Choose models ({selected.length}/5)</legend><div className="checks">{catalog.models.map((model) => <label key={model.id}><input type="checkbox" checked={selected.includes(model.id)} disabled={!selected.includes(model.id) && selected.length >= 5} onChange={() => toggle(model.id)} /> {model.display_name}</label>)}</div></fieldset>
        <fieldset><legend>Datasets ({selectedTaskIds.length}/{catalog.tasks.length})</legend><div className="checks">{catalog.tasks.map((task) => <label key={task.id}><input type="checkbox" checked={selectedTaskIds.includes(task.id)} disabled={selectedTaskIds.includes(task.id) && selectedTaskIds.length === 1} onChange={() => toggleTask(task.id)} /> {task.display_name}</label>)}</div></fieldset>
      </div>
      <details className="comparison-editor">
        <summary><span>Edit metrics</span><span className="comparison-summary">{selectionCount(selectedMetricCount, "metric")}</span></summary>
        <div className="comparison-controls">
          <section className="metric-filters" aria-labelledby="metric-filters-heading">
            <h2 id="metric-filters-heading">Metrics by dataset</h2>
            <p>Metric choices apply to every view in a dataset. Primary metrics are always shown.</p>
            <div className="metric-filter-grid">{catalog.tasks.filter((task) => selectedTaskIds.includes(task.id)).map((task) => {
              const lockedMetrics = primaryMetrics(task);
              return <fieldset key={task.id}><legend>{task.display_name}</legend>{taskMetrics(task).map((metric) => {
                const locked = lockedMetrics.includes(metric);
                return <label key={metric}><input type="checkbox" aria-label={`${metric} for ${task.display_name}`} checked={metricsByTask[task.id].includes(metric)} disabled={locked} onChange={() => toggleMetric(task.id, metric)} /> {metric}{locked ? " (primary)" : ""}</label>;
              })}</fieldset>;
            })}</div>
          </section>
        </div>
      </details>
      {selected.length < 2 ? <p className="empty">Select at least two models.</p> : <ComparisonTable catalog={catalog} rows={rows} selected={selected} selectedTaskIds={selectedTaskIds} metricsByTask={metricsByTask} />}
    </section>
  </>;
}

function taskMetrics(task: Task): string[] {
  return [...new Set(task.views.flatMap((view) => view.metrics))];
}

function primaryMetrics(task: Task): string[] {
  return [...new Set(task.views.map((view) => view.primary_metric))];
}

function selectionCount(count: number, singular: string): string {
  return `${count} ${singular}${count === 1 ? "" : "s"}`;
}

function ComparisonTable({ catalog, rows, selected, selectedTaskIds, metricsByTask }: { catalog: Catalog; rows: Result[]; selected: string[]; selectedTaskIds: string[]; metricsByTask: Record<string, string[]> }) {
  const models = new Map(catalog.models.map((model) => [model.id, model]));
  const visibleTasks = catalog.tasks.filter((task) => selectedTaskIds.includes(task.id));
  return <div className="table-wrap comparison-table-wrap"><table className="comparison-table">
    <thead><tr><th scope="col">Dataset / view</th>{selected.map((id) => {
      const model = models.get(id);
      return <th scope="col" key={id}><span className="model-header"><span className="model-name">{model?.display_name || id}</span>{model && <ModelInfo model={model} />}</span></th>;
    })}</tr></thead>
    {visibleTasks.map((task) => <tbody key={task.id} aria-label={task.display_name}>
      <tr className="dataset-heading"><th scope="rowgroup" colSpan={selected.length + 1}>{task.display_name}</th></tr>
      {task.views.map((view) => {
        const selectedMetrics = metricsByTask[task.id].filter((metric) => view.metrics.includes(metric));
        const metrics = [view.primary_metric, ...selectedMetrics.filter((metric) => metric !== view.primary_metric)];
        return <ComparisonRow key={view.id} view={view} metrics={metrics} results={rows.filter((row) => row.task_id === task.id && row.view_id === view.id)} selected={selected} />;
      })}
    </tbody>)}
  </table></div>;
}

function ComparisonRow({ view, metrics, results, selected }: { view: View; metrics: string[]; results: Result[]; selected: string[] }) {
  const bestScores = Object.fromEntries(metrics.map((metric) => [metric, Math.max(...results.map((result) => result.metrics[metric]).filter((score) => score !== undefined))]));
  return <tr className="view-row"><th scope="row"><span>{view.id}</span></th>{selected.map((id) => {
    const result = results.find((row) => row.model_id === id);
    if (!result) return <td key={id}><span className="missing">Missing result</span></td>;
    return <td key={id}><div className="result-cell"><div className="mini-metric-grid">{metrics.map((metric) => {
      const primary = metric === view.primary_metric;
      const isBest = result.metrics[metric] === bestScores[metric];
      const classes = ["mini-metric", primary ? "primary-metric" : "secondary-metric", isBest ? primary ? "primary-winner" : "secondary-winner" : ""].filter(Boolean).join(" ");
      return <div className={classes} data-metric={metric} key={metric}><span className="mini-metric-name">{metric}{primary && <span className="sr-only"> (primary metric)</span>}</span><span className="mini-score"><span className="sr-only">{isBest ? "Best score: " : "Score: "}</span>{fmt(result.metrics[metric])}</span></div>;
    })}</div>{result.status === "community" && <Badge status="community" />}</div></td>;
  })}</tr>;
}
