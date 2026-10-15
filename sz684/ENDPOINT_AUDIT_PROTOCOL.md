# Endpoint audit — October 15, 2026

Exploratory follow-up to the inspected transfer pilot. Run all three existing overlapping splits with the same model pair, selection-chosen layers, train-only PCA128 and alpha=1 ridge map. No tuning or new significance claims. Preserve existing outputs.

For 256 disjoint evaluation pairs per split, compare full mapped difference against target-endpoint-only prediction and negated source-endpoint-only prediction, using exactly the same 256 actual difference candidates. Also report affine endpoint retrieval among all 512 image endpoints (a different task/candidate count). Verify that differencing affine endpoint predictions equals mapping differences, and reproduce the first split's saved vectors.

Report full-dimensional cosine, top-1/top-5, target rank and projection retention. A 2D uncentered SVD basis fitted jointly to predicted and actual evaluation directions is for display only; scores never use projected coordinates. Joint display fitting is not an evaluation transformation. No semantic labels are inferred from arbitrary pairs.

Pair bootstrap intervals, if added later, must preserve endpoint pairing and describe conditional inference; repeated splits are not independent. This diagnostic uses descriptive statistics only.
