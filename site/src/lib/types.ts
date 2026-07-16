export type Status = "verified" | "community";

export interface TaskSubset { name: string; languages: string[]; }
export interface Task {
  name: string;
  display_name: string;
  description: string;
  type: string;
  main_score: string;
  dataset: { name: string; revision: string; url: string };
  splits: string[];
  subsets: TaskSubset[];
}

export interface Model {
  name: string;
  repository: string;
  revision: string;
  evaluated_at: string | null;
  status: Status;
  effective_prompts: Record<string, string>;
  n_parameters: number | "unknown";
  embed_dim: number | "unknown";
  is_latest: boolean;
}

export interface Result {
  model_name: string;
  model_revision: string;
  task_name: string;
  task_type: string;
  split: string;
  subset: string;
  languages: string[];
  metrics: Record<string, number>;
  main_score_name: string;
  main_score: number;
  dataset_name: string;
  dataset_revision: string;
  mteb_version: string;
  status: Status;
  result_path: string;
  result_sha256: string;
  effective_prompts: Record<string, string>;
  evaluated_at: string | null;
}

export interface Catalog {
  schema_version: 3;
  counts: { tasks: number; models: number; results: number };
  tasks: Task[];
  models: Model[];
  results: Result[];
}
