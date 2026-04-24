import { render, screen } from "@testing-library/react";
import { HeaderBar } from "./HeaderBar";

describe("HeaderBar", () => {
  it("renders title, subtitle, and listening status", () => {
    render(
      <HeaderBar
        title="Speech-to-Text"
        subtitle="Real-time Recognition"
        status="listening"
      />
    );

    expect(screen.getByRole("heading", { name: "Speech-to-Text" })).toBeInTheDocument();
    expect(screen.getByText("Real-time Recognition")).toBeInTheDocument();
    expect(screen.getByLabelText("Status: listening")).toBeInTheDocument();
  });

  it("applies the correct status class to the dot element", () => {
    const { container } = render(
      <HeaderBar
        title="Speech-to-Text"
        subtitle="Real-time Recognition"
        status="paused"
      />
    );

    const dot = container.querySelector(".status-pill__dot");
    expect(dot).toHaveClass("status-pill__dot--paused");
  });

  it("shows error status class for error state", () => {
    const { container } = render(
      <HeaderBar
        title="Speech-to-Text"
        subtitle="Real-time Recognition"
        status="error"
      />
    );

    const dot = container.querySelector(".status-pill__dot");
    expect(dot).toHaveClass("status-pill__dot--error");
  });
});
