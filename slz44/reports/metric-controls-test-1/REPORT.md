# Anisotropy controls under CKA, Gromov-Wasserstein cost and blind matching

Same 18 rows (3 splits x 6 model pairs), layers and fitted controls as the k-NN study. Excess = score minus shuffled-pairing mean; GW reduction = (null cost - observed cost) / null cost. Ranges are across dependent rows, not confidence intervals.

| Condition | k-NN excess mean [range]; up vs baseline /18 | CKA excess mean [range]; up vs baseline /18 | GW cost reduction mean [range]; up vs baseline /18 | GW blind-match top-1 mean [range]; up vs baseline /18 |
|---|---|---|---|---|
| baseline | 0.1048 [0.0914, 0.1275]; - | 0.2797 [0.1692, 0.3327]; - | 0.2288 [0.1446, 0.2997]; - | 0.0029 [0.0000, 0.0098]; - |
| center | 0.1171 [0.0971, 0.1445]; 18 | 0.2820 [0.1684, 0.3300]; 14 | 0.2452 [0.1736, 0.3163]; 16 | 0.0038 [0.0000, 0.0098]; 8 |
| pca128 | 0.1146 [0.0991, 0.1339]; 17 | 0.3018 [0.1851, 0.3488]; 18 | 0.2909 [0.1895, 0.3370]; 18 | 0.0034 [0.0000, 0.0137]; 6 |
| standardize | 0.1167 [0.0983, 0.1463]; 18 | 0.2660 [0.1739, 0.3194]; 1 | 0.2436 [0.1824, 0.3094]; 12 | 0.0029 [0.0000, 0.0078]; 8 |
| remove3 | 0.1119 [0.0897, 0.1496]; 14 | 0.1315 [0.0828, 0.1463]; 0 | 0.1337 [0.1039, 0.1624]; 0 | 0.0011 [0.0000, 0.0078]; 3 |
| pca128_whiten | 0.1001 [0.0716, 0.1349]; 7 | 0.0901 [0.0661, 0.1149]; 0 | 0.1078 [0.0795, 0.1362]; 0 | 0.0018 [0.0000, 0.0059]; 5 |

Blind-matching chance top-1 is 1/512 = 0.0020.

## Agreement between metrics

- Change vs baseline, k-NN excess vs CKA excess: Pearson r = +0.582 over 90 condition-rows (descriptive; rows are dependent).
- Change vs baseline, k-NN excess vs GW cost reduction: Pearson r = +0.426 over 90 condition-rows (descriptive; rows are dependent).
- Change vs baseline, CKA excess vs GW cost reduction: Pearson r = +0.937 over 90 condition-rows (descriptive; rows are dependent).

Exploratory. Protocol: METRIC_CONTROLS_PROTOCOL.md. Per-row values: scores.csv.
