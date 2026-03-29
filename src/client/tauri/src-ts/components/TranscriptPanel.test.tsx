import { render, screen } from "@testing-library/react";
import { TranscriptPanel } from "./TranscriptPanel";

describe("TranscriptPanel", () => {
  it("renders transcript text in a scrollable region", () => {
    render(
      <TranscriptPanel
        utterances={[
          { text: "Line one", start_time: 0, end_time: 1 },
          { text: "Line two", start_time: 1.2, end_time: 2 },
        ]}
        preliminaryText=""
        status="listening"
      />
    );

    const transcript = screen.getByRole("log", { name: "Transcript text" });

    expect(transcript).toBeInTheDocument();
    expect(transcript).toHaveTextContent("Line one");
    expect(transcript).toHaveTextContent("Line two");
  });

  it("shows status-appropriate empty state when no transcript text is available", () => {
    render(<TranscriptPanel utterances={[]} preliminaryText="" status="listening" />);
    expect(screen.getByText("Start speaking — transcript will appear here.")).toBeInTheDocument();
  });

  it("shows connecting message when status is connecting", () => {
    render(<TranscriptPanel utterances={[]} preliminaryText="" status="connecting" />);
    expect(screen.getByText("Connecting to recognition service…")).toBeInTheDocument();
  });

  it("shows paused message when status is paused", () => {
    render(<TranscriptPanel utterances={[]} preliminaryText="" status="paused" />);
    expect(screen.getByText("Recognition paused.")).toBeInTheDocument();
  });

  it("renders preliminary text in gray italic style", () => {
    render(
      <TranscriptPanel
        utterances={[{ text: "Finalized.", start_time: 0, end_time: 1 }]}
        preliminaryText="...still talking"
        status="listening"
      />
    );

    const preliminary = screen.getByText("...still talking");
    expect(preliminary).toBeInTheDocument();
    expect(preliminary).toHaveClass("transcript-panel__preliminary");
  });

  it("does not render preliminary span when preliminary text is empty", () => {
    render(
      <TranscriptPanel
        utterances={[{ text: "Done.", start_time: 0, end_time: 1 }]}
        preliminaryText=""
        status="listening"
      />
    );

    const transcript = screen.getByRole("log", { name: "Transcript text" });
    expect(transcript.querySelector(".transcript-panel__preliminary")).toBeNull();
  });

  it("hides empty state when only preliminary text is present", () => {
    render(<TranscriptPanel utterances={[]} preliminaryText="hearing something..." status="listening" />);

    expect(
      screen.queryByText("Start speaking — transcript will appear here.")
    ).not.toBeInTheDocument();
  });

  it("inserts space between utterances with small time gap", () => {
    render(
      <TranscriptPanel
        utterances={[
          { text: "Hello", start_time: 0, end_time: 1 },
          { text: "world", start_time: 1.5, end_time: 2.5 },
        ]}
        preliminaryText=""
        status="listening"
      />
    );

    const transcript = screen.getByRole("log", { name: "Transcript text" });
    expect(transcript).toHaveTextContent("Hello");
    expect(transcript).toHaveTextContent("world");
    expect(transcript.querySelectorAll("br")).toHaveLength(0);
  });

  it("shows connection error inside the surface when connectionError is set", () => {
    render(
      <TranscriptPanel
        utterances={[]}
        preliminaryText=""
        status="error"
        connectionError="Cannot connect to server. Is the STT server running?"
      />
    );

    const alert = screen.getByRole("alert");
    expect(alert).toBeInTheDocument();
    expect(alert).toHaveTextContent("Cannot connect to server");
    expect(
      screen.queryByText("Start speaking — transcript will appear here.")
    ).not.toBeInTheDocument();
  });

  it("inserts paragraph break between utterances with gap over 2 seconds", () => {
    render(
      <TranscriptPanel
        utterances={[
          { text: "First", start_time: 0, end_time: 1 },
          { text: "Second", start_time: 3.1, end_time: 4 },
        ]}
        preliminaryText=""
        status="listening"
      />
    );

    const transcript = screen.getByRole("log", { name: "Transcript text" });
    expect(transcript.querySelectorAll("br")).toHaveLength(2);
    expect(transcript).toHaveTextContent("First");
    expect(transcript).toHaveTextContent("Second");
  });
});
