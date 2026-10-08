import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import App from "./App";

const fetchMock = vi.fn();
global.fetch = fetchMock;

const predictionResponse = {
  canonical_smiles: "CCO",
  predicted_log_s: -0.123,
  descriptors: {
    MolWt: 46.07,
    LogP: -0.1,
    TPSA: 20.23,
    HBD: 1,
    HBA: 1,
    RotBonds: 0,
  },
  svg: "<svg xmlns=\"http://www.w3.org/2000/svg\"><g data-testid=\"molecule-svg\" /></svg>",
  disclaimer: "This prediction is an estimate.",
};

describe("Molecular Property Predictor", () => {
  beforeEach(() => {
    fetchMock.mockReset();
  });

  it("submits a SMILES string and displays the prediction", async () => {
    const user = userEvent.setup();
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => predictionResponse,
    });

    render(<App />);
    await user.type(screen.getByLabelText("SMILES"), "CCO");
    await user.click(screen.getByRole("button", { name: "Predict LogS" }));

    expect(fetchMock).toHaveBeenCalledWith("http://127.0.0.1:8003/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ smiles: "CCO" }),
    });
    expect(await screen.findByText("Predicted aqueous solubility")).toBeInTheDocument();
    expect(screen.getByText("-0.123")).toBeInTheDocument();
    expect(screen.getByText("LogS (log₁₀ mol/L)")).toBeInTheDocument();
    expect(screen.getByText("Canonical SMILES")).toBeInTheDocument();
  });

  it("populates the selected example and shows its API prediction", async () => {
    const user = userEvent.setup();
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({ ...predictionResponse, canonical_smiles: "CC(=O)Oc1ccccc1C(=O)O" }),
    });

    render(<App />);
    const aspirinButton = screen.getByRole("button", { name: /Aspirin/i });
    await user.click(aspirinButton);

    expect(screen.getByLabelText("SMILES")).toHaveValue("CC(=O)Oc1ccccc1C(=O)O");
    expect(aspirinButton).toHaveAttribute("aria-pressed", "true");

    await user.click(screen.getByRole("button", { name: "Predict LogS" }));

    expect(await screen.findByText("-0.123")).toBeInTheDocument();
    expect(screen.getByTestId("molecule-svg")).toBeInTheDocument();
  });

  it("shows the loading state while a prediction is pending", async () => {
    const user = userEvent.setup();
    let resolveRequest;
    fetchMock.mockImplementation(() => new Promise((resolve) => {
      resolveRequest = resolve;
    }));

    render(<App />);
    await user.type(screen.getByLabelText("SMILES"), "CCO");
    await user.click(screen.getByRole("button", { name: "Predict LogS" }));

    expect(screen.getByRole("button", { name: "Predicting…" })).toBeDisabled();

    resolveRequest({
      ok: true,
      json: async () => predictionResponse,
    });
    await waitFor(() => expect(screen.getByText("-0.123")).toBeInTheDocument());
  });

  it("shows a request error when prediction fails", async () => {
    const user = userEvent.setup();
    fetchMock.mockResolvedValue({
      ok: false,
      json: async () => ({ detail: "Invalid SMILES." }),
    });

    render(<App />);
    await user.type(screen.getByLabelText("SMILES"), "invalid");
    await user.click(screen.getByRole("button", { name: "Predict LogS" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Invalid SMILES.");
  });
});
