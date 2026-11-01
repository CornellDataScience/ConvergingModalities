# Methods

Geometry compares balanced concept centroids: average within original source,
then sources equally. All models share one sorted concept roster. No learned
cross-modal map enters geometry evaluation.

- RSA correlates strict upper triangles of cosine-distance matrices using
  Spearman ranks. Matrix cells are dependent; ordinary Spearman p-values are not used.
- Neighbor metrics use top-k cosine neighbors, exclude self, and break ties by
  row order. Jaccard and intersection/k are different metrics. This is labeled
  graph agreement, not graph isomorphism.
- Linear CKA uses centered, unnormalized pooled features. It supports different
  widths and is invariant to orthogonal rotations and isotropic scaling.
- Whole-concept correspondence shuffles preserve internal geometry. One-sided
  p-values are exploratory, unadjusted, and assume exchangeable concepts. Related
  concepts can violate that assumption. No confirmatory claims follow directly.

Directions divide training families into separate anchor and probe groups.
Train-only PCA gives a common Euclidean space; Procrustes fits on shared anchor
centroids. Balanced ridge binary classifiers define CAVs, with label 1 always the
target state. Select alpha from 0.1, 1, 10 using validation balanced accuracy.
Test families never select parameters. Both transfer directions are evaluated.

With row vectors and `target ≈ source @ R`, orthogonal normal transport is
`normal @ R`. The classifier's plane origin must also be transported. General
ridge maps do not transport normals by the displacement rule. These CAVs are
not mean endpoint differences or full TCAV output-derivative measurements.

Operational retrieval steps along a unit mapped CAV. Scale selection uses
validation AP, includes zero, and chooses the smaller scale on ties. Exclude the
query's source from candidates. Relevant targets match endpoint label and family.
Report family-macro AP, identity retention, and joint success. Controls include
no shift, reverse, and a single norm-matched random direction. Fits, transforms,
probe parameters, anchor IDs, selected scales, skipped relations and scores are saved.

Current directional inference is fixed-fit point estimation, not refit-inclusive
confidence intervals. Random ensembles, wrong-relation controls, split-half
reliability, cluster uncertainty, block-aware nulls, multiplicity adjustment,
power analysis and independent-source replication remain research work. Three
recordings per concept do not characterize a stable concept distribution.

High geometry similarity does not prove identical semantics or successful
identity-preserving transfer. Legacy data are weakly labeled and previously
inspected. New AV data lack verified directional endpoints.

References: [RSA](https://doi.org/10.3389/neuro.06.004.2008),
[CKA](https://proceedings.mlr.press/v97/kornblith19a.html),
[CAV/TCAV](https://proceedings.mlr.press/v80/kim18d.html),
[VGGSounder](https://github.com/Bizilizi/VGGSounder),
[BEATs](https://github.com/microsoft/unilm/tree/master/beats).
