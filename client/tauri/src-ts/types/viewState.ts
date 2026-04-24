export type HeaderStatus = "connecting" | "waiting" | "listening" | "paused" | "error";

export interface ModelInfo {
  name: string;
  display_name: string;
  size_description: string;
  status: string;
}

export interface DownloadProgress {
  model_name: string;
  status: string;
  progress?: number;
  downloaded_bytes?: number;
  total_bytes?: number;
  error_message?: string;
}

export interface FinalizedUtterance {
  text: string;
  start_time: number;
  end_time: number;
}

export interface AppViewState {
  title: string;
  subtitle: string;
  status: HeaderStatus;
  utterances: FinalizedUtterance[];
  preliminaryText: string;
  isPaused: boolean;
  connectionError?: string;
  serverState?: string;
  models: ModelInfo[];
  downloadProgress?: DownloadProgress;
  downloadDialogDismissed?: boolean;
}
