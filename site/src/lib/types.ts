export type Status = "verified" | "community";

export interface View {
  id: string;
  split: string;
  languages: string[];
  metrics: string[];
  primary_metric: string;
}

export interface Task {
  id: string;
  version: number;
  display_name: string;
  description: string;
  dataset: { id: string; revision: string; url?: string };
  adapter: string;
  views: View[];
}

export interface Model {
  id: string;
  display_name: string;
  hf_id: string;
  revision: string;
  languages: string[];
  homepage?: string;
  parameter_count: number | "unknown";
  vocab_size: number | "unknown";
}

export interface Result {
  model_id: string;
  task_id: string;
  view_id: string;
  metrics: Record<string, number>;
  status: Status;
  model_revision: string;
  dataset_revision: string;
  parameter_count: number;
  vocab_size: number;
}

export interface Catalog {
  tasks: Task[];
  models: Model[];
  results: Result[];
}
