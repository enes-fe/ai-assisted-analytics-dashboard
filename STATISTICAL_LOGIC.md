# Statistical Logic Scope

This project keeps the default dashboard intentionally narrow. The thesis-facing claim is not "fully automated statistical inference"; it is an AI-assisted analytics dashboard where semantic selection is assisted by AI and all calculations are performed deterministically in backend code.

## Core Supported Analysis

- Dataset profiling: column types, missingness, cardinality, numeric summaries, categorical summaries, and preview data.
- KPI generation: 3 to 5 concise KPI cards, with deterministic aggregation in Pandas-backed services.
- Chart selection: 4 to 6 main charts by default, using supported renderers only: bar, line, area, pie, donut, scatter, table, forecast, and clustering.
- Forecasting: generated only when a valid date column and numeric time series are available. Forecast output includes fit/backtest warnings, and weak fit is described cautiously.
- Clustering: generated only when at least two usable numeric features and enough rows are available. Silhouette and overlap diagnostics are shown, and weak segmentation is described as exploratory.
- PDF report export: exports the visible KPI, chart, forecast, clustering, insight, and warning sections.

## Advanced / Experimental Statistical Capabilities

The following code is retained because it is useful and working, but it is not part of the default dashboard narrative:

- Normality checks.
- Mann-Whitney and Kruskal-Wallis group tests.
- Fisher exact and Chi-Square categorical tests.
- FDR-adjusted q-values.
- Post-hoc pair summaries.
- Effect sizes and effect labels.
- Significance-driven boxplots, regression overlays, and Cramer's V metadata.

These capabilities live primarily in `backend/services/stat_engine.py` and the advanced branch of `backend/services/chart_generator.py`. They are available through the debug/backward-compatible endpoint:

```text
/api/analytics/core/{dataset_id}?advanced=true
```

The default core analytics endpoint keeps `advanced_statistics.enabled=false` and does not run or return advanced statistical tests.

## Known Limitations

- The dashboard should not be interpreted as a causal inference system. Insight text should describe associations, distributions, forecasts, or exploratory segments, not causes.
- Segment names are descriptive labels inferred from feature patterns. They are not best/worst rankings and should be validated with domain context.
- Forecasts are simple model outputs with limited diagnostics. Low R2, high backtest error, or low data volume means the forecast is a cautious scenario rather than a precise prediction.
- Clustering quality depends on feature choice and data geometry. Low silhouette or high overlap means segmentation is exploratory.
- Heatmap and treemap requests may be detected in prompts, but they are not current supported renderers. The app should fall back to a supported chart and state that the requested renderer is unavailable.
- Advanced statistical tests are sensitive to sample size, missingness, multiple testing, and distribution assumptions. They are best treated as future or debug capabilities unless explicitly enabled and reviewed.

## Future Improvements

- Add an explicit advanced/debug UI toggle before exposing advanced test cards in the application.
- Add renderer support only when heatmap or treemap charts are actually implemented and verified.
- Add stronger eligibility summaries for forecasting and clustering before model execution.
- Expand report methodology notes if advanced statistics become part of the defended thesis scope.
