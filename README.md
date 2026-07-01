# AI-Assisted Automatic Analytics Dashboard

A full-stack analytics dashboard that turns uploaded tabular datasets into a focused BI-style dashboard. The project was built as a thesis/portfolio application with a React frontend, a FastAPI backend, deterministic Pandas analytics, and an optional LLM planning layer.

The core product idea is deliberately narrow: AI can help identify semantic column roles and analysis intent, but it does not calculate metrics. KPI values, chart data, grouping, sorting, filtering, forecasting, clustering, and statistical outputs are produced by backend code.

## Features

- Upload CSV, JSON, XLS, or XLSX files.
- Preview and paginate uploaded datasets.
- Generate KPI cards and charts from backend analytics.
- Build prompt-driven charts through the chat endpoint.
- Run forecast and clustering flows when the dataset is eligible.
- Export dashboard reports as PDF.
- Switch between English and Turkish UI text.
- Keep advanced statistical tests behind opt-in/debug paths.

## Tech Stack

- Frontend: Vite, React, TypeScript, Recharts, lucide-react.
- Backend: FastAPI, Pandas, SQLAlchemy, scikit-learn, SciPy, statsmodels.
- Storage: local SQLite metadata plus parquet/pickle dataset files under `data_storage/`.
- AI: Groq-backed semantic planning for the primary fast dashboard path, with legacy/local Ollama helpers retained for development/debug flows.

## Project Structure

```text
src/                         React frontend
src/components/              Dashboard, upload, chart, KPI, export UI
backend/main.py              FastAPI app and API routes
backend/services/            Deterministic analytics, charts, ML, stats
backend/services/ai/         LLM planning and validation layer
backend/.env.example         Safe local configuration template
scratch/                     Smoke and regression scripts
sample.csv                   Small demo dataset
AI_SETUP.md                  AI provider setup notes
STATISTICAL_LOGIC.md         Statistical scope and limitations
```

## Quick Start

Prerequisites:

- Node.js 20 or newer
- Python 3.10 or newer

Install frontend dependencies:

```powershell
npm install
```

Create the backend environment file:

```powershell
Copy-Item backend\.env.example backend\.env
```

Install backend dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

Start the backend API from the `backend` directory:

```powershell
cd backend
uvicorn main:app --reload --port 8000
```

In a second terminal, start the frontend:

```powershell
npm run dev
```

Open the Vite URL, usually `http://localhost:5173`. The frontend proxies `/api` requests to `http://localhost:8000`.

## AI Configuration

The app can run without a real API key: upload and preview flows remain available, while AI dashboard generation returns a clear safe error. To enable the primary AI path, edit `backend/.env`:

```text
AI_PROVIDER=groq
GROQ_API_KEY=your_real_key_here
GROQ_MODEL=llama-3.1-8b-instant
AI_SEMANTIC_TIMEOUT_SECONDS=8
AI_CHAT_TIMEOUT_SECONDS=8
```

See [AI_SETUP.md](AI_SETUP.md) for the full environment-variable list and validation notes.

## Validation

Useful checks before publishing or demoing:

```powershell
npm run lint
npm run build
python -m compileall backend/services backend/main.py
python scratch/test_backend_functions.py
python scratch/test_api_endpoints.py
python scratch/statistical_engine_evaluation.py
```

Some smoke scripts expect the local Python environment and backend dependencies to be installed.

## Release Hygiene

- Local secrets stay in `backend/.env`, which is ignored.
- Local datasets, databases, logs, virtual environments, `node_modules/`, and `dist/` are ignored.
- `backend/.env.example` is safe to commit and should be copied for local setup.
- `sample.csv` is a small synthetic demo dataset.

## Scope Notes

- The dashboard is exploratory analytics software, not a causal inference system.
- Forecasts and clusters are shown only when the dataset has enough usable structure.
- Weak forecasts, weak clustering, missing AI credentials, or unsupported chart requests should produce cautious messages instead of misleading charts.
- Advanced statistical tests are retained for debug/future work and are not the default dashboard claim.
