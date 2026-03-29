export type HeaderStatus = "connecting" | "listening" | "paused" | "error";

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
}
