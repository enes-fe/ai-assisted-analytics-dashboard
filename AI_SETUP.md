# Local Ollama AI Setup

The AI layer uses a local Ollama server to produce structured JSON chart plans.
It never calculates metrics directly; backend Pandas code still performs all
grouping, aggregation, sorting, filtering, and `chartData` generation.

## Prerequisite

Ollama should already be installed on the developer machine.

Pull the default local model manually:

```bash
ollama pull qwen2.5-coder:3b
```

Optional stronger model:

```bash
ollama pull qwen2.5-coder:7b
```

By default, the backend expects Ollama at:

```text
http://localhost:11434
```

## Environment Variables

```text
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5-coder:3b
AI_SEMANTIC_ENABLED=true
AI_CHAT_ENABLED=true
AI_MAX_COLUMN_CARDS=60
AI_MAX_RECOMMENDED_CHARTS=8
```

If Ollama or the selected model is unavailable, the app should not crash.
`/api/analytics/core/{dataset_id}` falls back to heuristic charts, and
`/api/chat` falls back to the existing fuzzy prompt parser.

## Manual Checks

1. Football dataset with `Player`, `Team`, `Position`, `Goals`, `Assists`,
   `Minutes`, `Rating`.
   Expected: `detected_domain` mentions football/soccer/player performance,
   primary metrics include `Goals` and `Assists`, and recommended charts include
   top goals by player, top assists by player, and goals vs assists scatter.

2. Sales dataset with `Product`, `Region`, `Sales`, `Profit`, `Quantity`,
   `Order_Date`.
   Expected: `detected_domain` mentions sales, primary metrics include `Sales`
   and `Profit`, and recommended charts include sales by region/product plus a
   date trend when `Order_Date` exists.

3. Chat prompt: `gol ve asistleri oyuncuya göre göster`.
   Expected: the AI plan uses goals/assists and player/team where available,
   instead of unrelated columns such as age or minutes as the primary metric.

4. Ollama unavailable.
   Expected: the app keeps working, `analytics/core` returns `heuristic_only`,
   and `/api/chat` returns `fallback_heuristic`.
