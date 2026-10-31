# State directions: text-derived change vs held-out visual change

Cosine between the bridged text direction (training categories) and the visual difference between states in held-out categories. Margin = mean matching-relation cosine minus mean other-relation cosine; top-1 chance = 0.20. Fold-averaged; ranges are over 5 dependent folds. Shuffled-bridge null: 200 shuffles.

| Condition | Text margin [fold range] | Text top-1 | Shuffled-bridge margin mean / 95th pct | p | Image-derived margin | Image top-1 | Stability text / image |
|---|---|---|---|---|---|---|---|
| reference | 0.360 [0.253, 0.463] | 0.68 | -0.006 / 0.078 | 0.005 | 0.328 | 0.64 | 0.99 / 0.31 |
| uncentered_pca64 | 0.353 [0.246, 0.465] | 0.72 | 0.005 / 0.085 | 0.005 | 0.333 | 0.64 | 1.00 / 0.32 |
| center_only | 0.283 [0.174, 0.386] | 0.68 | 0.002 / 0.061 | 0.005 | 0.273 | 0.72 | 0.99 / 0.26 |
| standardize_pca64 | 0.368 [0.271, 0.456] | 0.76 | -0.002 / 0.073 | 0.005 | 0.340 | 0.68 | 0.99 / 0.33 |
| remove3_pca64 | 0.313 [0.217, 0.401] | 0.64 | -0.002 / 0.068 | 0.005 | 0.349 | 0.80 | 1.00 / 0.30 |
| whiten_pca64 | 0.273 [0.185, 0.377] | 0.76 | -0.000 / 0.050 | 0.005 | 0.297 | 0.88 | 1.00 / 0.23 |

## Alignment matrix, reference (rows: text direction; columns: held-out visual direction)

| | cleaning | cooking | ripening | thawing | drying |
|---|---|---|---|---|---|
| cleaning | 0.396 | 0.024 | 0.016 | 0.107 | 0.071 |
| cooking | 0.041 | 0.419 | -0.091 | 0.562 | -0.036 |
| ripening | 0.058 | -0.136 | 0.400 | -0.184 | 0.107 |
| thawing | 0.100 | 0.405 | -0.123 | 0.628 | -0.021 |
| drying | 0.127 | -0.017 | 0.054 | -0.023 | 0.217 |

Exploratory. Protocol: STATE_DIRECTIONS_PROTOCOL.md. Groups of images, not before-and-after pairs.
