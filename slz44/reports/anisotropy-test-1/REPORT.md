# Anisotropy-control sensitivity

Three overlapping splits; six model pairs; N=512, k=10. Layers and clipping fixed from selection halves; every control fitted on selection rows only. Each condition uses the same 1,000 matched permutations. Ranges are across dependent comparisons, not confidence intervals.

| Condition | Raw range | Excess range | Mean change vs baseline | Increases / 18 | Strict size increases / 6 | Mean cosine (lang / vision) | Effective rank (lang / vision) | Top-PC share (lang / vision) |
|---|---|---|---:|---:|---:|---|---|---|
| baseline | 0.1109–0.1471 | 0.0913–0.1275 | 0.000000 | 0 | 2 | 0.364 / 0.476 | 399.192 / 260.626 | 0.059 / 0.149 |
| center | 0.1164–0.1639 | 0.0969–0.1444 | 0.012153 | 18 | 1 | 0.002 / 0.004 | 400.291 / 262.598 | 0.059 / 0.162 |
| pca128 | 0.1186–0.1533 | 0.0990–0.1338 | 0.009787 | 17 | 2 | 0.002 / 0.004 | 105.625 / 95.170 | 0.093 / 0.183 |
| standardize | 0.1178–0.1656 | 0.0983–0.1462 | 0.011816 | 18 | 1 | 0.002 / 0.004 | 403.482 / 272.431 | 0.056 / 0.137 |
| remove3 | 0.1094–0.1689 | 0.0898–0.1494 | 0.007042 | 14 | 4 | 0.002 / 0.003 | 411.493 / 285.902 | 0.036 / 0.082 |
| pca128_whiten | 0.0912–0.1543 | 0.0717–0.1347 | -0.004709 | 7 | 5 | 0.002 / 0.003 | 120.192 / 118.931 | 0.026 / 0.027 |

## Matched differences

- standardize minus baseline: mean 0.011816; range 0.002344 to 0.024023; increases in 18/18 comparisons.
- remove3 minus baseline: mean 0.007042; range -0.006055 to 0.021875; increases in 14/18 comparisons.
- pca128_whiten minus baseline: mean -0.004709; range -0.024219 to 0.009375; increases in 7/18 comparisons.
- pca128_whiten minus pca128 (whitening alone): mean -0.014497; range -0.030273 to 0.000977; increases in 1/18 comparisons.
- remove3 minus center (top-direction removal alone): mean -0.005111; range -0.014648 to 0.006055; increases in 3/18 comparisons.

Stored condition scores above their corresponding null 95th percentile: 108/108.

Diagnostics are means over the 18 rows, where each model appears in several rows; they describe geometry and do not show that anisotropy caused a score change. Baseline scores and all null draws match the earlier held-out run. Layers are baseline-selected, not each control's best. PCA and whitening are fitted per modality; whitening is inside the 128-dimensional subspace because the 512-row selection covariance is rank deficient. Per-dimension scales from 512 rows are noisy. No claims about N>512, independent replications, or statistically established differences.

Protocol: ANISOTROPY_PROTOCOL.md. Reproduce: `python anisotropy.py --baseline <run> --heldout <results.json>`. Full draws/provenance: results.json; detailed rows: scores.csv.
