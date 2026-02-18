## AI Co-Scientist Challenge 2026  
### Data Minimization vs. Model Performance Trade-off

This repository provides a reproducible experimental pipeline for our submission
to the AI Co-Scientist Challenge 2026, evaluating the trade-off between data
minimization and model performance on the Adult dataset.

We analyze how four minimization strategies (H, C, P, R) impact accuracy and
F1-score across multiple models. Strategy-specific LLM choices are intentional:
GPT is used for semantically sensitive strategies (P, R), while Copilot is used
for the structurally deterministic C strategy to reduce stochastic variance.

A replication of the vertical data minimization baseline (Staab et al., 2023)
is included to explicitly contrast feature generalization with our staged
feature-removal design space.

**Project Structure**

- **Main entry**: `experiments/main_results/run_final_experiment.py`
- **Reference replication (Staab et al., 2023)**: `experiments/replication/staab2023_vdm/run_replication_iterative.py`
- **Comparison**: `docs/differences_vs_reference.md`
- **Paper artifacts**: `paper/` (manuscript, final figures, tables)
- **Legacy code**: `experiments/main_results/legacy/` (reference only)
  
All figures and tables reported in the paper are generated from the main entry script.
