## Differences vs. Staab et al. 2023 (Vertical Data Minimization)

- Reference (Staab2023 vDM): minimize by **generalizing feature granularity** (bucketing), not removing columns. :contentReference[oaicite:10]{index=10}
- Our work: treat **feature-level removal order** as a design space (H/P/C/R) and measure how it reshapes **error profiles** (e.g., recall collapse).

| Dimension | Staab2023 vDM | Our pipeline |
|---|---|---|
| Minimization operation | Generalization (buckets) | Removal (staged feature groups) |
| Primary focus | Utility–privacy trade-off w/ adversaries | Utility + error profile shift under compliance-style removal |
| Key risk highlighted | empirical privacy risk (breach adversaries) | silent failure modes (recall collapse) |
