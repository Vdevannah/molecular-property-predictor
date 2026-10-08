import React from "react";
import { render, screen, waitFor, within } from "@testing-library/react";
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
  molecular_formula: "C2H6O",
  experimental_reference: {
    status: "available",
    experimental_log_s: -0.1,
    absolute_error: 0.023,
    esol_source: "ESOL (Delaney) dataset",
    compound_id: "ETOH",
    phase3_split: "training",
  },
  applicability_domain: {
    status: "Within descriptor ranges",
    warnings: [],
    descriptor_ranges: [
      { descriptor: "MolWt", minimum: 18.0, maximum: 450.0 },
    ],
    training_molecules: 902,
    source: "Phase 3 random train split",
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
    expect(screen.getByTestId("prediction-value")).toHaveTextContent("-0.123");
    expect(screen.getByTestId("scientific-predicted-log-s")).toHaveTextContent("-0.123");
    expect(screen.getByText("LogS (log₁₀ mol/L)")).toBeInTheDocument();
    expect(screen.getByText("Canonical SMILES")).toBeInTheDocument();
    expect(screen.getByText("Experimental vs. Predicted Solubility")).toBeInTheDocument();
    const formula = screen.getByText("Molecular formula").nextElementSibling;
    expect(formula).toHaveTextContent("C₂H₆O");
    expect(formula).not.toHaveTextContent("C2H6O");
    expect(screen.getByText("Within descriptor ranges")).toBeInTheDocument();
  });

  it("formats the aspirin formula with subscripts", async () => {
    const user = userEvent.setup();
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({ ...predictionResponse, molecular_formula: "C9H8O4" }),
    });

    render(<App />);
    await user.type(screen.getByLabelText("SMILES"), "CC(=O)Oc1ccccc1C(=O)O");
    await user.click(screen.getByRole("button", { name: "Predict LogS" }));

    expect(await screen.findByText("C₉H₈O₄")).toBeInTheDocument();
  });

  it("formats the ibuprofen formula with subscripts", async () => {
    const user = userEvent.setup();
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({ ...predictionResponse, molecular_formula: "C13H18O2" }),
    });

    render(<App />);
    await user.type(screen.getByLabelText("SMILES"), "CC(C)Cc1ccc(C(C)C(=O)O)cc1");
    await user.click(screen.getByRole("button", { name: "Predict LogS" }));

    expect(await screen.findByText("C₁₃H₁₈O₂")).toBeInTheDocument();
  });

  it("formats the strychnine formula with subscripts", async () => {
    const user = userEvent.setup();
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({ ...predictionResponse, molecular_formula: "C21H22N2O2" }),
    });

    render(<App />);
    await user.type(screen.getByLabelText("SMILES"), "CCO");
    await user.click(screen.getByRole("button", { name: "Predict LogS" }));

    expect(await screen.findByText("C₂₁H₂₂N₂O₂")).toBeInTheDocument();
  });

  it("shows an explicit unavailable experimental reference", async () => {
    const user = userEvent.setup();
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        ...predictionResponse,
        experimental_reference: {
          status: "not_available",
          experimental_log_s: null,
          absolute_error: null,
          esol_source: null,
          compound_id: null,
          phase3_split: "not_in_phase3_split",
        },
      }),
    });

    render(<App />);
    await user.type(screen.getByLabelText("SMILES"), "CCN");
    await user.click(screen.getByRole("button", { name: "Predict LogS" }));

    expect(await screen.findByText("No experimental reference available in ESOL.")).toBeInTheDocument();
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

    expect(screen.getByTestId("prediction-value")).toHaveTextContent("-0.123");
    expect(screen.getByTestId("molecule-svg")).toBeInTheDocument();
  });

  it("clears the selected example when the input does not match an example", async () => {
    const user = userEvent.setup();

    render(<App />);
    const aspirinButton = screen.getByRole("button", { name: /Aspirin/i });
    await user.click(aspirinButton);
    expect(aspirinButton).toHaveAttribute("aria-pressed", "true");

    await user.clear(screen.getByLabelText("SMILES"));
    await user.type(screen.getByLabelText("SMILES"), "CCN");

    expect(screen.getByLabelText("SMILES")).toHaveValue("CCN");
    expect(aspirinButton).toHaveAttribute("aria-pressed", "false");
  });

  it("keeps the selected example synchronized with the input SMILES", async () => {
    const user = userEvent.setup();

    render(<App />);
    const quickExamples = screen.getByRole("region", { name: "Quick-select examples" });
    const ethanolButton = within(quickExamples).getByRole("button", { name: /Ethanol/i });
    await user.click(ethanolButton);

    const input = screen.getByLabelText("SMILES");
    await user.clear(input);
    await user.type(input, "CCO");

    expect(ethanolButton).toHaveAttribute("aria-pressed", "true");
  });

  it("filters and selects a molecule from the searchable catalog", async () => {
    const user = userEvent.setup();

    render(<App />);
    const searchInput = screen.getByRole("searchbox", { name: "Search molecules" });
    const searchResults = screen.getByRole("region", { name: "Molecule search results" });
    await user.type(searchInput, "acid");

    expect(screen.getByRole("button", { name: /Acetic acid/i })).toBeInTheDocument();
    expect(within(searchResults).getAllByRole("button")).toHaveLength(2);

    await user.click(screen.getByRole("button", { name: /Acetic acid/i }));

    expect(screen.getByLabelText("SMILES")).toHaveValue("CC(=O)O");
    expect(screen.getByRole("button", { name: /Acetic acid/i })).toHaveAttribute("aria-pressed", "true");
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
    await waitFor(() => expect(screen.getByTestId("prediction-value")).toHaveTextContent("-0.123"));
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
