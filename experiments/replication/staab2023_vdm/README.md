## Replication: Staab et al. (2023) Vertical Data Minimization (Iterative baseline)

We implement a lightweight replication of the *Iterative* vDM baseline from
Staab et al., “From Principle to Practice: Vertical Data Minimization for Machine Learning” (arXiv:2311.10500). :contentReference[oaicite:7]{index=7}

Unlike our main paper (feature *removal* as a staged design space),
vDM performs feature *generalization* (bucketization) to reduce data granularity. :contentReference[oaicite:8]{index=8}

Implementation details follow the paper’s Iterative description:
(1) discrete values are sorted by logistic-regression weights and grouped into k buckets via dynamic programming,
(2) continuous values are split into k-quantiles,
(3) bucket counts are reduced while classifier error remains under a threshold T. :contentReference[oaicite:9]{index=9}

Entry point: `run_replication_iterative.py`  
Outputs: `results_staab2023_iterative_adult.csv`
