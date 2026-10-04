# Paired difference uncertainty: protocol fixed before this analysis

## Why the evaluation changes

The prior transductive metric uses every evaluation point as both query and neighbor candidate. Ordinary query bootstrapping would ignore shared-pool dependence; a naive full bootstrap also introduces duplicate points into nearest-neighbor graphs. We therefore use a separate fixed reference bank. This is an inductive variant of the measurement, not a confidence interval for the exact earlier transductive statistic.

For each of the three saved splits, keep the 512 selection rows and their selected layers/clipping/transforms. Divide the ordered evaluation half into 256 reference pairs and 256 query pairs. Fit preprocessing only on selection rows. Recompute query-to-reference neighbors for every condition. Shared query pairs are the sampling units. Use existing baseline, centering, and centered PCA128 definitions.

## Six contrasts per split

All effects are raw overlap differences (positive favors the first condition):
1. Centered minus baseline, average over six fixed model pairs.
2. PCA128 minus baseline, average over six fixed model pairs.
3. PCA128 minus centered, average over six fixed model pairs.
4. BLOOMZ 1.7B minus 560M at baseline, average over the two fixed vision models (each model pair retains its selection-chosen layers).
5. Reference pool 256 minus 128 at baseline, fixed k=10, averaged over six pairs.
6. Reference pool 256/k=10 minus 128/k=5 at baseline, approximately fixed neighbor fraction, averaged over six pairs.

Use the first 128 reference rows as the nested smaller bank. Save raw pool differences and also subtract the analytical random-overlap difference (10/256-10/128 for fixed k, zero for fixed fraction). The latter is the primary pool contrast because random overlap changes with pool size. Here queries are not reference candidates, so chance is k/R, not k/(N-1).

Compute per-query paired differences before averaging across queries. Models are a fixed roster, not independent experimental replicates. Use 20,000 paired query bootstrap resamples (resampling the already-computed difference is algebraically equivalent to using the same query indices for both conditions), BCa intervals via scipy.stats.bootstrap. Report 95% intervals and 99.7222% intervals (1 - .05/18) to Bonferroni-adjust the family of six contrasts times three splits. Dependence between these intervals does not invalidate Bonferroni, but bootstrap coverage is approximate. Do not use bootstrap sign frequencies as p-values.

Primary conclusion rules: an adjusted interval excluding zero supports a conditional difference for that split. Claim consistency across reference choices only if signs agree and all three adjusted intervals exclude zero. These are overlapping exploratory splits, not three independent replications. Conditioning excludes uncertainty in reference sampling, layer selection, and transform fitting. Query sampling assumes exchangeable image-caption pairs; unmodeled source/topic clusters may invalidate that assumption. The data were already inspected, so these are exploratory intervals, not prospective confirmation.

Save exact query/reference row IDs, per-query scores and differences, bootstrap distributions, source hashes, detailed results, and plot. Check interval code and reference scoring against independent calculations. Do not overwrite previous studies.
