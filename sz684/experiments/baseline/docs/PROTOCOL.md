# Frozen reduced experiment protocol

Established before inspecting the 1024-pair alignment results.

- Original WIT-1024 release, in original order; all 1024 paired examples.
- BLOOMZ 560M, 1.1B, 1.7B; DINOv2 small, base. Exact IDs and immutable revisions in each manifest.
- Float32, MPS inference, batch size 1, no quantization or truncation.
- The released default clipping, per-layer L2 normalization, k=10, maximum over individual layer pairs; CPU float32 scoring.
- 100 correspondence permutations, seed 20240927, with the same full layer search repeated for every permutation. These test non-random alignment, not the scaling hypothesis.
- Reduced OpenWebText evaluation: sorted first 64 documents from the linked OpenWebText-10k archive, up to 1024 characters each. Byte-weighted NLL, EOS context, at most 256 preceding context tokens and 256 target tokens. Every text token is scored exactly once. Corpus text and archive SHA-256 are retained.

## Descriptive checks

For each of the two vision models, inspect whether alignment increases with measured language performance across the three language checkpoints. Also report the larger-versus-smaller vision comparison at each fixed language checkpoint. Report reversals rather than dropping them. Pearson correlations are descriptive only with n=3 related checkpoints. No significant-correlation claim or universal-convergence claim is made.

An observed direction consistent with the paper is a reduced-scope qualitative agreement. A reversal is a reduced-scope discrepancy, not a falsification of the original full experiment. The dataset size is matched, but model roster, environment, precision, and language evaluation differ. Unknown original layer concatenation and checkpoint naming remain unresolved. Without matched numerical reference arrays, numerical reproduction remains unassessed.

## Validation gate

- Independent set-intersection metric check, self-neighbor exclusion, identity and joint-permutation invariance.
- Permutation graph relabeling matches explicit recomputation.
- Both clipping paths match their upstream definitions.
- Padding-aware average pooling and ordered cache rows checked.
- Target-window accounting covers every language token exactly once.
- First pilot pair's saved features match the upstream-style extraction path at atol=rtol=1e-5 (observed maximum difference 0 for both models).

## Later experiments

No tuning to match an expected curve. Preserve this baseline before changes to sample size, k, clipping, pooling, model family, or held-out layer selection. Record any engineering correction separately from methodological changes.
