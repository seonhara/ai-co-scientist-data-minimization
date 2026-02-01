## AI Co-Scientist Challenge 2026  
### Data Minimization vs. Model Performance Trade-off

This repository provides a reproducible experimental pipeline for our submission
to the AI Co-Scientist Challenge 2026, evaluating the trade-off between data
minimization and model performance on the Adult dataset.

We analyze how four minimization strategies (H, C, P, R) impact accuracy and
F1-score across multiple models. Strategy-specific LLM choices are intentional:
GPT is used for semantically sensitive strategies (P, R), while Copilot is used
for the structurally deterministic C strategy to reduce stochastic variance.

**Main entry**: `experiments/main_results/run_final_experiment.py`  
**Paper artifacts**: `paper/` (manuscript, final figures, tables)  
**Legacy code**: `experiments/main_results/legacy/` (reference only)

All figures and tables reported in the paper are generated from the main entry script.
