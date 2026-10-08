# Phase 5 Teach-Back

## Overview

Phase 5 turns the molecular-property pipeline into a small full-stack application. The React interface collects a SMILES input, sends it to FastAPI, displays the calculated descriptors, and renders the resulting molecule as an SVG.

## React components, props, and state

React components are reusable functions that return JSX. The root component is the application, while child elements such as buttons, cards, and descriptor articles are rendered from arrays.

The main component stores the current SMILES value, selected example, prediction result, loading state, and error message in React state.

```jsx
const [smiles, setSmiles] = useState("");
const [selectedExample, setSelectedExample] = useState(null);
const [result, setResult] = useState(null);
const [error, setError] = useState("");
const [loading, setLoading] = useState(false);
```

The React state values drive what the user sees. For example, `loading` disables the prediction button and changes its label, while `result` determines whether the prediction panel is rendered.

## Event handling and controlled inputs

The SMILES textarea is a controlled input. Its value always comes from React state, and each change calls `setSmiles`.

Example selection uses a button click handler:

```jsx
function handleExampleSelect(example) {
  setSelectedExample(example.name);
  setSmiles(example.smiles);
  setError("");
}
```

The prediction button sends the current state to the API. Because the input is controlled, editing the SMILES manually replaces the selected example and creates a new request when the user submits the form.

## Fetch API and asynchronous requests

The frontend uses the Fetch API to send a JSON POST request:

```jsx
const response = await fetch(`${API_BASE_URL}/api/predict`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ smiles }),
});
```

`await` pauses execution until the network request finishes. When the request has not yet finished, the button shows a loading label. When the response is successful, the JSON payload is stored in React state. When the response is unsuccessful, the server detail is displayed as an error.

## FastAPI routes and Pydantic validation

The backend exposes a health route at `/api/health` and a prediction route at `/api/predict`.

The request model requires a non-empty SMILES string:

```python
class PredictRequest(BaseModel):
    smiles: str = Field(..., min_length=1)
```

The field validator removes surrounding whitespace and uses RDKit to test whether the string describes a valid molecule. The canonicalized molecule is returned to the client.

The response model defines the contract returned by FastAPI:

- canonical SMILES
- predicted LogS
- calculated descriptors
- RDKit SVG
- scientific disclaimer

## CORS and local development ports

The frontend runs on `http://localhost:5175`, while the backend runs on `http://127.0.0.1:8003`.

FastAPI uses CORS middleware so the browser can make requests from the local React origin. The frontend reads its API endpoint from `VITE_API_BASE_URL`, with the default set to `http://127.0.0.1:8003`.

## RDKit descriptors and SVG rendering

RDKit converts each SMILES string into a molecule object and calculates six descriptors:

1. MolWt
2. LogP
3. TPSA
4. HBD
5. HBA
6. RotBonds

The descriptors are passed to the saved Random Forest model in the exact trained column order.

RDKit also produces a two-dimensional SVG representation of the molecule. The SVG is inserted into the React page with `dangerouslySetInnerHTML`, while CSS constrains the drawing area and preserves its aspect ratio. This keeps small molecules readable and larger molecules within their container without stretching the drawing.

## Loading the trained Random Forest model

The FastAPI application loads one saved model artifact and one feature-order artifact when the application starts. The model is kept in memory for subsequent requests.

The server verifies:

- the saved model exists;
- the feature list exists;
- the feature list is not empty;
- the model feature count matches the saved list;
- the saved feature order matches the expected training order.

If the model files are unavailable, the prediction endpoint returns HTTP 503 instead of failing unpredictably.

## Complete SMILES-to-prediction workflow

```text
User enters SMILES
  -> FastAPI validates and canonicalizes the string
  -> RDKit creates the molecule
  -> Six descriptors are calculated
  -> Descriptor values are ordered to match training
  -> Random Forest predicts LogS
  -> RDKit generates a 2D SVG
  -> FastAPI returns JSON
  -> React displays prediction and structure
```

## Scientific limitations

The Random Forest model was trained on the ESOL dataset and learns statistical relationships between the selected descriptors and measured LogS.

The model does not provide experimental measurements, and predictions may be less reliable outside the chemical space represented by ESOL. The application therefore emphasizes that predicted values are estimates rather than validated measurements.

## Interview questions

### 1. How does React keep the input and displayed value synchronized?

React state is the source of truth. The textarea uses a value controlled by state, while handlers update that state when the user edits the input.

### 2. Why use FastAPI Pydantic models?

Pydantic validates incoming data and returns a consistent response schema. The backend can reject missing or invalid input before calculation begins.

### 3. Why is feature order important?

The saved Random Forest model expects descriptors in a specific sequence. Reordering the values would change the meaning of the feature vector and produce incorrect predictions.

### 4. Why is CORS required for local development?

The React development server and FastAPI backend run on different ports. Browsers enforce the same-origin policy, so CORS explicitly permits the React origin to access the API.

### 5. What is the difference between the model prediction and the molecular structure?

The model predicts LogS from descriptor values. The RDKit SVG is a visualization of the molecule and does not itself generate the prediction; it is produced from the same validated SMILES input.

## Two-minute technical project explanation

This project builds a machine-learning model that predicts aqueous solubility from molecular structure. RDKit converts a SMILES string into a molecule and calculates six descriptors: molecular weight, partition coefficient, polar surface area, hydrogen-bond donors, hydrogen-bond acceptors, and rotatable bonds.

The model is trained on the ESOL dataset and saved with its feature order. In the full-stack application, FastAPI validates the input, calculates descriptors, runs the trained Random Forest, and returns the prediction with a molecular SVG. React displays the structure, prediction, descriptor values, and scientific limitations.

The interface includes example molecules and supports local development over ports 5175 and 8003. The application does not retrain or alter the saved model during inference.
