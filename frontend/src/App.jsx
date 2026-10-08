import React, { useMemo, useState } from "react";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8003";

const EXAMPLE_MOLECULES = [
  { name: "Ethanol", smiles: "CCO" },
  { name: "Benzene", smiles: "c1ccccc1" },
  { name: "Aspirin", smiles: "CC(=O)Oc1ccccc1C(=O)O" },
  { name: "Caffeine", smiles: "Cn1c(=O)c2c(ncn2C)n(C)c1=O" },
  { name: "Methanol", smiles: "CO" },
  { name: "Acetone", smiles: "CC(=O)C" },
  { name: "Acetic acid", smiles: "CC(=O)O" },
  { name: "Toluene", smiles: "Cc1ccccc1" },
  { name: "Phenol", smiles: "Oc1ccccc1" },
  { name: "Aniline", smiles: "Nc1ccccc1" },
  { name: "Cyclohexane", smiles: "C1CCCCC1" },
  { name: "Naphthalene", smiles: "c1ccc2ccccc2c1" },
  { name: "Ibuprofen", smiles: "CC(C)Cc1ccc(C(C)C(=O)O)cc1" },
  { name: "Benzoic acid", smiles: "O=C(O)c1ccccc1" },
  { name: "Acetaminophen", smiles: "CC(=O)Nc1ccc(O)cc1" },
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
const formatError = (value) => value === null || Number.isNaN(value) ? "—" : `${formatValue(value)}`;
const subscriptDigits = Object.fromEntries(Array.from({ length: 10 }, (_, index) => [String(index), String.fromCodePoint(0x2080 + index)]));
const formatFormula = (formula) => formula.replace(/[0-9]/g, (digit) => subscriptDigits[digit]);

function MolecularIcon() {
  return (
    <svg viewBox="0 0 64 64" aria-hidden="true">
      <circle cx="32" cy="32" r="5" className="atom-core" />
      <circle cx="18" cy="19" r="4.5" className="atom-node" />
      <circle cx="46" cy="18" r="4.5" className="atom-node" />
      <circle cx="16" cy="45" r="4.5" className="atom-node" />
      <circle cx="48" cy="46" r="4.5" className="atom-node" />
      <path d="M22 22 L30 30 M42 22 L34 30 M20 41 L30 34 M44 41 L34 34" />
    </svg>
  );
}

export default function App() {
  const [smiles, setSmiles] = useState("");
  const [searchTerm, setSearchTerm] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const selectedExample = useMemo(
    () => EXAMPLE_MOLECULES.find(
      (example) => example.smiles.trim() === smiles.trim(),
    )?.name ?? null,
    [smiles],
  );

  const searchableExamples = useMemo(() => {
    const term = searchTerm.trim().toLowerCase();
    const quickNames = new Set(EXAMPLE_MOLECULES.slice(0, 4).map((example) => example.name));
    return EXAMPLE_MOLECULES.filter((example) => !quickNames.has(example.name)
      && (example.name.toLowerCase().includes(term)
        || example.smiles.toLowerCase().includes(term)));
  }, [searchTerm]);

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
    setSmiles(example.smiles);
    setError("");
  }

  function handleSmilesChange(event) {
    setSmiles(event.target.value);
    setError("");
  }

  return (
    <main className="page-shell">
      <div className="app-glow app-glow-one" aria-hidden="true" />
      <div className="app-glow app-glow-two" aria-hidden="true" />

      <header className="topbar">
        <div className="brand-wrap">
          <div className="brand-mark" aria-hidden="true">
            <MolecularIcon />
          </div>
          <div>
            <p className="eyebrow">Cheminformatics platform</p>
            <h1>Molecular Intelligence</h1>
          </div>
        </div>
        <div className="header-meta">
          <span className="tech-badge">RDKit + Machine Learning</span>
          <p className="header-description">
            AI-Powered Molecular Property Prediction
          </p>
        </div>
      </header>

      <section className="hero" aria-labelledby="hero-title">
        <div className="hero-copy">
          <p className="section-kicker">Molecular Insight</p>
          <h2 id="hero-title">From Molecular Structure to Predictive Insight</h2>
          <p>
            Explore aqueous solubility predictions powered by molecular descriptors
            and machine learning.
          </p>
          <div className="hero-metrics" aria-label="Scientific platform summary">
            <div>
              <strong>1,128</strong>
              <span>ESOL molecules</span>
            </div>
            <div>
              <strong>6</strong>
              <span>Descriptors</span>
            </div>
            <div>
              <strong>1</strong>
              <span>ML model</span>
            </div>
          </div>
        </div>
        <div className="hero-network" aria-hidden="true">
          <svg viewBox="0 0 360 220">
            <path d="M34 151 L96 78 L168 118 L237 67 L326 137" />
            <path d="M96 78 L128 180 L168 118 L228 154 L326 137" />
            <path d="M34 151 L128 180 L237 67" />
            <circle cx="34" cy="151" r="6" />
            <circle cx="96" cy="78" r="6" />
            <circle cx="168" cy="118" r="6" />
            <circle cx="237" cy="67" r="6" />
            <circle cx="128" cy="180" r="6" />
            <circle cx="228" cy="154" r="6" />
            <circle cx="326" cy="137" r="6" />
          </svg>
        </div>
      </section>

      <section className="workspace" aria-label="Molecular prediction workspace">
        <form className="input-panel" onSubmit={handleSubmit}>
          <div className="panel-header">
            <div>
              <p className="section-kicker">Input workspace</p>
              <h3>Enter SMILES</h3>
            </div>
            <span className="panel-chip">Canonicalized</span>
          </div>

          <label htmlFor="smiles" className="sr-only">SMILES</label>
          <textarea
            id="smiles"
            value={smiles}
            onChange={handleSmilesChange}
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
              <h3 id="examples-title">Quick select</h3>
            </div>
            <div className="example-grid" role="region" aria-label="Quick-select examples">
              {EXAMPLE_MOLECULES.slice(0, 4).map((example) => (
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

          <div className="search-section" aria-labelledby="search-title">
            <div className="search-header">
              <h3 id="search-title">Molecule catalog</h3>
            </div>
            <label htmlFor="molecule-search" className="search-label">Search molecules</label>
            <input
              id="molecule-search"
              type="search"
              value={searchTerm}
              onChange={(event) => setSearchTerm(event.target.value)}
              placeholder="Search by name or SMILES"
              aria-label="Search molecules"
            />
            <div className="search-results" role="region" aria-label="Molecule search results">
              {searchableExamples.length > 0 ? searchableExamples.map((example) => (
                <button
                  key={example.name}
                  type="button"
                  className={`search-result ${selectedExample === example.name ? "selected" : ""}`}
                  aria-pressed={selectedExample === example.name}
                  onClick={() => handleExampleSelect(example)}
                >
                  <span>{example.name}</span>
                  <small>{example.smiles}</small>
                </button>
              )) : <p className="no-results">No matching molecules found.</p>}
            </div>
          </div>

          <button type="submit" className="primary-button" disabled={loading || !smiles.trim()}>
            <span className="button-icon" aria-hidden="true">↗</span>
            {loading ? "Predicting…" : "Predict Solubility"}
          </button>
          {error ? <p className="message error" role="alert">{error}</p> : null}
        </form>

        <aside className="info-panel" aria-labelledby="logS-title">
          <div className="info-panel-header">
            <span className="info-kicker">Property</span>
            <h3 id="logS-title">LogS</h3>
          </div>
          <p className="logS-definition">
            The base-10 logarithm of aqueous molar solubility. Higher values generally
            indicate greater predicted solubility.
          </p>
          <div className="info-metric">
            <span>Prediction</span>
            <strong>Machine learning estimate</strong>
          </div>
          <p className="scientific-note">Not an experimental measurement.</p>
          <div className="capability-list" aria-label="Platform capabilities">
            <span>RDKit descriptors</span>
            <span>Random Forest</span>
            <span>Scientific context</span>
          </div>
        </aside>
      </section>

      {result ? (
        <section className="result-panel" aria-live="polite">
          <div className="prediction-card">
            <div className="prediction-topline">
              <span className="section-kicker section-kicker-light">Prediction</span>
              <span className="signal-dot">Live model output</span>
            </div>
            <div className="prediction-value-wrap">
              <div className="prediction-value" data-testid="prediction-value">
                {formatValue(result.predicted_log_s)}
              </div>
              <div className="prediction-label">log₁₀(mol/L)</div>
            </div>
            <p className="prediction-explanation">
              Estimated aqueous solubility from the trained molecular descriptor model.
            </p>
            <p className="prediction-note">Model prediction — not an experimental measurement.</p>
          </div>

          <div className="science-grid">
            <article className="scientific-card identity-card">
              <div className="section-heading-row">
                <div>
                  <p className="section-kicker">Molecular identity</p>
                  <h3>Structure and composition</h3>
                </div>
                <span className="status-pill positive">RDKit structure</span>
              </div>

              <div className="structure-card">
                <div className="structure-svg-frame">
                  <div
                    className="structure-svg"
                    aria-label="Molecular structure"
                    dangerouslySetInnerHTML={{ __html: result.svg }}
                  />
                </div>
              </div>

              <div className="identity-detail">
                <div>
                  <span>Canonical SMILES</span>
                  <code>{result.canonical_smiles}</code>
                </div>
                <div>
                  <span>Molecular formula</span>
                  <strong>{formatFormula(result.molecular_formula)}</strong>
                </div>
              </div>
            </article>

            <article className="scientific-card validation-card">
              <div className="section-heading-row">
                <div>
                  <p className="section-kicker">Scientific validation</p>
                  <h3>Experimental vs. predicted solubility</h3>
                </div>
                <span className="split-label">
                  {result.experimental_reference.phase3_split === "training"
                    ? "Phase 3 training"
                    : result.experimental_reference.phase3_split === "testing"
                      ? "Phase 3 testing"
                      : "Not in Phase 3 split"}
                </span>
              </div>

              <div className="scientific-grid">
                <div className="scientific-stat">
                  <span>Predicted LogS</span>
                  <strong data-testid="scientific-predicted-log-s">{formatValue(result.predicted_log_s)}</strong>
                </div>
                <div className="scientific-stat">
                  <span>Experimental LogS</span>
                  <strong>{result.experimental_reference.experimental_log_s === null ? "—" : formatValue(result.experimental_reference.experimental_log_s)}</strong>
                </div>
                <div className="scientific-stat">
                  <span>Absolute error</span>
                  <strong>{result.experimental_reference.absolute_error === null ? "—" : formatError(result.experimental_reference.absolute_error)}</strong>
                </div>
                <div className="scientific-stat">
                  <span>ESOL source</span>
                  <strong>{result.experimental_reference.esol_source || "Unavailable"}</strong>
                </div>
                <div className="scientific-stat">
                  <span>Train/test status</span>
                  <strong>{result.experimental_reference.phase3_split === "training" ? "Training set" : result.experimental_reference.phase3_split === "testing" ? "Held-out test set" : "Not in Phase 3 split"}</strong>
                </div>
                <div className="scientific-stat">
                  <span>Applicability</span>
                  <strong>{result.applicability_domain.status}</strong>
                </div>
              </div>

              <div className="experimental-detail">
                {result.experimental_reference.status === "not_available" ? (
                  <p>No experimental reference available in ESOL.</p>
                ) : result.experimental_reference.status === "conflicting_measurements" ? (
                  <p>
                    Multiple ESOL measurements exist for this canonical SMILES: {result.experimental_reference.conflicting_measurements.map((value) => formatValue(value)).join("; ")}. The record is ambiguous and no value was selected.
                  </p>
                ) : (
                  <p>
                    Experimental value from {result.experimental_reference.esol_source} for {result.experimental_reference.compound_id}.
                  </p>
                )}
              </div>

              <div className="domain-block">
                <div className="domain-header">
                  <h4>Descriptor-range assessment</h4>
                  <span className={result.applicability_domain.status === "Within descriptor ranges" ? "domain-good" : "domain-warning"}>{result.applicability_domain.status}</span>
                </div>
                <p>
                  Training-set descriptor ranges: {result.applicability_domain.training_molecules} molecules.
                  {result.applicability_domain.warnings.length > 0
                    ? ` Warning(s): ${result.applicability_domain.warnings.join(", ")}.`
                    : " No descriptor warnings."}
                </p>
                {result.applicability_domain.warnings.length > 0 ? (
                  <ul className="warning-list">
                    {result.applicability_domain.warnings.map((descriptor) => (
                      <li key={descriptor}>{descriptor} is outside the Phase 3 training range.</li>
                    ))}
                  </ul>
                ) : null}
                <p className="domain-note">Within descriptor ranges does not guarantee reliable prediction. This is not calibrated confidence or prediction uncertainty.</p>
              </div>
            </article>
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
              This model uses six calculated descriptors and does not measure experimental
              solubility. Predictions may be less reliable for chemical series outside the
              training data or for molecules with unusual interactions.
            </p>
          </div>
        </section>
      ) : null}
    </main>
  );
}
