export type DataRow = Record<string, string | number | boolean | null>;

export interface UploadResponse {
  dataset_id: number;
  filename: string;
  data: DataRow[];
  columns: string[];
  row_count: number;
}

export interface DatasetSelection extends UploadResponse {
  reset?: false;
}

export interface ResetSelection {
  reset: true;
}

export type PendingSelection = DatasetSelection | ResetSelection;

export interface ApiErrorDetail {
  code?: string;
  message?: string;
}
