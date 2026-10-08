import React, { useMemo, useState } from "react";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8003";

const EXAMPLE_MOLECULES = [
  { name: "Ethanol", smiles: "CCO" },
  { name: "Benzene", smiles: "c1ccccc1" },
  { name: "Aspirin", smiles: "CC(=O)Oc1ccccc1C(=O)O" },
  { name: "Caffeine", smiles: "Cn1c(=O)c2c(ncn2C)n(C)c1=O" },
];

const DESCRIPTOR_LABELS = {
  MolWt: "Molecular weight",
  LogP: "Octanol/water partition coefficient",
  TPSA: "Topological polar surface area",
  HBD: "Hydrogen-bond donors",
  HBA: "Hydrogen-bond acceptors",
  RotBonds: "Rotatable bonds",
};

const formatValue = (value) => Number(value).toFixed(3);

export default function App() {
  const [smiles, setSmiles] = useState("");
  const [selectedExample, setSelectedExample] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const descriptorCards = useMemo(
    () => (result ? Object.entries(result.descriptors) : []),
    [result],
  );

  async function handleSubmit(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles }),
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Prediction request failed.");
      }
      setResult(payload);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  function handleExampleSelect(example) {
    setSelectedExample(example.name);
    setSmiles(example.smiles);
    setError("");
  }

  return (
    <main className="page-shell">
      <header className="topbar">
        <div className="brand-wrap">
          <div className="brand-mark" aria-hidden="true">M</div>
          <div>
            <p className="eyebrow">Cheminformatics</p>
            <h1>Molecular Property Predictor</h1>
          </div>
        </div>
        <p className="header-description">
          Estimate aqueous solubility from molecular structure using RDKit descriptors
          and a trained Random Forest model.
        </p>
      </header>

      <section className="hero">
        <form className="input-panel" onSubmit={handleSubmit}>
          <label htmlFor="smiles" className="field-label">SMILES</label>
          <textarea
            id="smiles"
            value={smiles}
            onChange={(event) => setSmiles(event.target.value)}
            placeholder="e.g. CCO"
            rows="4"
            spellCheck={false}
            aria-describedby="smiles-help"
          />
          <p id="smiles-help" className="help-text">
            Enter a valid SMILES string to generate descriptors and predict LogS.
          </p>

          <div className="examples-section" aria-labelledby="examples-title">
            <div className="examples-header">
              <h2 id="examples-title">Try an example molecule</h2>
            </div>
            <div className="example-grid">
              {EXAMPLE_MOLECULES.map((example) => (
                <button
                  key={example.name}
                  type="button"
                  className={`example-button ${selectedExample === example.name ? "selected" : ""}`}
                  aria-pressed={selectedExample === example.name}
                  onClick={() => handleExampleSelect(example)}
                >
                  <span>{example.name}</span>
                  <small>{example.smiles}</small>
                </button>
              ))}
            </div>
          </div>

          <button type="submit" disabled={loading || !smiles.trim()}>
            {loading ? "Predicting…" : "Predict LogS"}
          </button>
          {error ? <p className="message error" role="alert">{error}</p> : null}
        </form>

        <aside className="info-panel">
          <p className="info-title">What is LogS?</p>
          <p>
            LogS is the base-10 logarithm of aqueous molar solubility. Higher values
            generally indicate greater predicted solubility.
          </p>
          <p className="scientific-note">
            Machine learning estimate — not an experimental measurement.
          </p>
        </aside>
      </section>

      {result ? (
        <section className="result-panel" aria-live="polite">
          <div className="prediction-card">
            <div>
              <p className="eyebrow">Prediction</p>
              <h2>Predicted aqueous solubility</h2>
            </div>
            <div className="prediction-value-wrap">
              <div className="prediction-value" data-testid="prediction-value">
                {formatValue(result.predicted_log_s)}
              </div>
              <div className="prediction-label">LogS (log₁₀ mol/L)</div>
            </div>
            <p className="prediction-explanation">
              LogS is the base-10 logarithm of aqueous molar solubility. Higher values
              generally indicate greater predicted solubility.
            </p>
            <p className="prediction-note">Machine learning estimate — not an experimental measurement.</p>
          </div>

          <div className="structure-section">
            <div className="structure-card">
              <h3>2D structure</h3>
              <div className="structure-svg-frame">
                <div
                  className="structure-svg"
                  dangerouslySetInnerHTML={{ __html: result.svg }}
                />
              </div>
            </div>
            <div className="summary-card">
              <h3>Canonical SMILES</h3>
              <code>{result.canonical_smiles}</code>
              <p className="disclaimer">{result.disclaimer}</p>
              <p className="dataset-note">
                Trained on the ESOL dataset; predictions may be less reliable outside its
                training domain.
              </p>
            </div>
          </div>

          <div className="descriptor-grid">
            {descriptorCards.map(([name, value]) => (
              <article key={name} className="descriptor-card">
                <span>{DESCRIPTOR_LABELS[name]}</span>
                <strong>{formatValue(value)}</strong>
                <small>{name}</small>
              </article>
            ))}
          </div>

          <div className="limitations">
            <h3>Scientific limitations</h3>
            <p>
              This model uses six calculated descriptors and does not measure
              experimental solubility. Predictions may be less reliable for chemical
              series outside the training data or for molecules with unusual
              interactions.
            </p>
          </div>
        </section>
      ) : null}
    </main>
  );
}
