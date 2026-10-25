# State-transfer preprocessing sensitivity

Own sample of MIT-States, five relations, five category-held-out folds. Only the preprocessing differs between conditions. Intervals are descriptive category-label bootstrap intervals (2,000 draws, conditional on per-category means), not adjusted for the six conditions.

| Condition | AP no change | AP shared | AP gain [95%] | Category kept, no change | Category kept, shared | Change [95%] | Joint no change | Joint shared | Mean cosine (image / text) |
|---|---:|---:|---|---:|---:|---|---:|---:|---|
| reference | 0.3692 | 0.4338 | +0.0646 [+0.0342, +0.0982] | 0.856 | 0.765 | -0.0910 [-0.1621, -0.0252] | 0.234 | 0.416 | 0.031 / 0.024 |
| uncentered_pca64 | 0.3592 | 0.4317 | +0.0725 [+0.0418, +0.1078] | 0.858 | 0.774 | -0.0847 [-0.1483, -0.0238] | 0.235 | 0.424 | 0.358 / 0.370 |
| center_only | 0.3762 | 0.4300 | +0.0538 [+0.0224, +0.0855] | 0.884 | 0.804 | -0.0796 [-0.1551, -0.0085] | 0.262 | 0.430 | 0.023 / 0.040 |
| standardize_pca64 | 0.3703 | 0.4370 | +0.0667 [+0.0337, +0.1030] | 0.856 | 0.750 | -0.1054 [-0.1757, -0.0393] | 0.245 | 0.418 | 0.031 / 0.021 |
| remove3_pca64 | 0.3620 | 0.4259 | +0.0639 [+0.0278, +0.1030] | 0.852 | 0.762 | -0.0898 [-0.1643, -0.0211] | 0.237 | 0.402 | 0.036 / 0.018 |
| whiten_pca64 | 0.3695 | 0.4295 | +0.0600 [+0.0240, +0.0959] | 0.878 | 0.791 | -0.0871 [-0.1558, -0.0217] | 0.259 | 0.387 | 0.030 / 0.043 |

## All methods under the reference condition

| Method | AP | First result has both | State | Category |
|---|---:|---:|---:|---:|
| no_change | 0.3692 | 0.234 | 0.305 | 0.856 |
| shared_text | 0.4338 | 0.416 | 0.591 | 0.765 |
| image_derived | 0.4323 | 0.398 | 0.565 | 0.761 |
| reversed | 0.2932 | 0.146 | 0.186 | 0.805 |
| random | 0.3673 | 0.247 | 0.318 | 0.841 |
| wrong_change | 0.3848 | 0.288 | 0.364 | 0.839 |
| direct_text | 0.4028 | 0.375 | 0.714 | 0.548 |

## Limits

Exploratory. Same data for all conditions; folds share categories and are not independent replications; the sample is a fresh draw, so numbers are not expected to match any earlier sample exactly, only the pattern (shared direction above no change, reversed below, category retention below no change). Layers were chosen in an earlier, unrelated study. A condition chosen after seeing these results would need a fresh evaluation. Diagnostics describe geometry and do not show cause.

Protocol: STATE_CONTROLS_PROTOCOL.md. Per-category values and chosen step sizes: results.json.
