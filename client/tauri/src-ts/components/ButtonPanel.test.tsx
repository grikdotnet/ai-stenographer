import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import { ButtonPanel } from "./ButtonPanel";

describe("ButtonPanel", () => {
  it("renders required controls and handles callbacks", async () => {
    const user = userEvent.setup();
    const onPauseToggle = vi.fn();
    const onClear = vi.fn();

    render(
      <ButtonPanel
        isPaused={false}
        onPauseToggle={onPauseToggle}
        onClear={onClear}
      />
    );

    await user.click(screen.getByRole("button", { name: "Pause" }));
    await user.click(screen.getByRole("button", { name: "Clear" }));

    expect(screen.getByLabelText("Controls")).toBeInTheDocument();
    expect(onPauseToggle).toHaveBeenCalledTimes(1);
    expect(onClear).toHaveBeenCalledTimes(1);
  });

  it("shows Resume when paused", () => {
    render(
      <ButtonPanel
        isPaused={true}
        onPauseToggle={vi.fn()}
        onClear={vi.fn()}
      />
    );

    expect(screen.getByRole("button", { name: "Resume" })).toBeInTheDocument();
  });
});
