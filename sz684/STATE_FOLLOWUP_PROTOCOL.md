# Exploratory preservation follow-up

Designed after primary natural-state results were inspected. Primary results and analysis code remain unchanged. This follow-up cannot be relabeled confirmatory merely because it uses validation data for its parameter choices.

Observed primary finding: text transfer improves state/noun AP against all four adjusted controls but loses noun-top1 retrieval, failing the predeclared preservation guardrail. Test two remedies:

1. Constrain the original scale search to validation noun-top1 losses no worse than .05.
2. Estimate an object-identity proxy subspace from training-noun image centroids, retaining80% of between-centroid squared singular values. Remove0%,50% or100% of a relation direction's component in that subspace, then norm-match to the original relation. Select projection strength and scale jointly using the same validation noun guardrail and AP objective. Include strength0 and scale0; break ties toward smaller strength then smaller scale.

Training centroids are not proven disentangled object identity directions. This projection is a diagnostic method, not a novelty claim. Assess AP versus both no shift and the original direction, and noun retention versus no shift, on the same frozen test queries. Report ordinary95% paired clustered intervals descriptively with no new adjusted significance claims. Retain every validation trial and selected choice. A positive exploratory outcome requires a fresh verified-label replication before a strong method claim.
