# Track 1 / M2 protocol: bridge controls, held-out retrieval, alignment vs retrieval gain

Status: **draft for Week 1 freeze.** Committed before any MIT-States result has been computed with this code. Any change made after seeing test results must be listed under "Post-freeze changes" at the bottom, with the reason.

## Question

When a text state direction (for example "cooked chicken" − "raw chicken", averaged over training objects) is mapped into image space with the existing bridge, does adding it to a held-out image retrieve the **same object in the target state** better than controls? And does a direction's agreement with the visual direction (M1's alignment matrix) predict how much retrieval it buys?

## Inputs (contract until M1's schema lands)

| File | Contents | Owner |
|---|---|---|
| `bridge.npz` | `text` (n×d_t), `image` (n×d_i): paired rows used to fit the bridge. Default: the 512 WIT selection pairs of split 20261004, BLOOMZ-1.7B / DINOv2-base at sz684's selected layers, after sz684's clipping. | sz684 cache |
| `states.npz` | `image_X`, `image_object`, `image_state` (one row per MIT-States image); `text_X`, `text_object`, `text_state` (one row per prompt; several templates per (object, state) are averaged). Same models, layers and clipping as the bridge. | M3 data, M1 extraction |
| `split.json` | `train` / `val` / `test` object lists (family-level split, disjoint), and `transitions`: `[{"name", "source", "target"}]`. | M3 (frozen test set) |

Preprocessing is fixed to the existing bridge: L2-normalize, PCA-128 fitted **on bridge rows only** per modality, L2-normalize. Ridge map text→image, α_ridge = 1, intercept via centering. Nothing is fitted on MIT-States test objects.

## Directions (all fitted on `train` objects only)

For transition s→t, eligible objects are train objects that have text for both states and at least one image in both states.

- **text direction**: mean over eligible objects of text(o, t) − text(o, s), mapped through the bridge.
- **visual direction**: mean over eligible objects of mean image(o, t) − mean image(o, s).

All directions are unit-normalized before use.

## Conditions

| Condition | Query for source image x | What it controls for |
|---|---|---|
| `no_change` | x | Baseline: retrieval with no edit |
| `bridge` | x + λ·W(text direction) | The method under test |
| `shuffled_bridge` | same, W refit on shuffled bridge pairs (200 draws) | Does the bridge's pairing matter, or does any linear map do? |
| `reversed_bridge` | x − λ·W(text direction) | Is the sign meaningful? |
| `other_transition` | x + λ·mean of the other transitions' mapped directions | Is the direction specific to this state change, or a generic "processed" shift? |
| `subset_a`, `subset_b` | text direction from two disjoint random halves of the eligible train objects | Does the result depend on which objects built the direction? |
| `image_direction` | x + λ·visual direction | Reference: what a direction learned directly in image space buys (not a guaranteed ceiling) |

Queries are normalized after the shift. λ comes from the fixed grid {0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2}, chosen by maximizing the primary metric on `val` objects. There is one global λ per condition and no per-transition tuning. λ = 0 is excluded from the grid, and `no_change` is the λ = 0 reference.

- `bridge`, `shuffled_bridge` (per draw), `subset_a`, `subset_b` and `image_direction` each get their own validated λ, so every candidate method gets the same tuning budget.
- `reversed_bridge` and `other_transition` **reuse the bridge's λ** ("same step, wrong direction"). If they were tuned themselves, they would retreat to the smallest step and behave like `no_change`, which makes the specificity test vacuous. The synthetic null showed this.

## Retrieval

The gallery for a partition is every image of that partition's objects, in every state. A query is a source-state image whose object has at least one target-state image in the gallery. The query image itself is excluded from its own ranking.

- **Primary metric:** `joint_rr`, the reciprocal rank of the first gallery image with the **same object and the target state**.
- **Secondary metrics:** `joint_at1`, `joint_at5`, `object_at1` (object retention: is the top-1 image the same object?), and `state_at1`.

Object retention is always reported next to joint success on the same queries. A gain in state retrieval that comes from losing the object is the failure mode Track 2 is about.

## Inference

The test-set contrasts are paired per query, on the primary metric:

1. bridge − no_change
2. bridge − shuffled_bridge (per-query mean over draws)
3. bridge − reversed_bridge
4. bridge − other_transition
5. image_direction − bridge (descriptive gap to the reference)

**Uncertainty:** percentile bootstrap over **source objects** (cluster resampling, 10,000 draws). Images of the same object are not independent. Contrasts 1 to 4 form the confirmatory family and get Bonferroni-adjusted intervals (95% → 1 − 0.05/4). Contrast 5 is descriptive.

**Shuffled-bridge tail fraction:** (1 + #draws ≥ observed) / 201 on mean test `joint_rr`. The layers were picked using true WIT correspondences, so the shuffle doesn't repeat the whole selection procedure (same caveat as sz684's TRANSFER_PROTOCOL). It is therefore not reported as a p-value.

**Decision rule.** Transfer is "supported" only if (a) all four confirmatory adjusted intervals lie above 0 **and** (b) the shuffled-bridge tail fraction is ≤ 0.05. Both conditions are needed:

- The query bootstrap holds the fitted bridge fixed, so it can't see that the bridge is one random draw. In the null world, a lucky bridge passes (a) alone.
- `bridge − no_change` is not specific by itself. Any shift moves the query off its own source images, which can raise `joint_rr` from its near-floor `no_change` value.

The rule was checked on the synthetic world before any real data: 9/9 seeds with signal pass, 9/9 null seeds fail.

## Alignment vs retrieval gain

- **Alignment matrix:** A[i, j] = cos(mapped text direction i, visual direction j), both from train objects. The **specificity margin** for transition i is A[i, i] − mean over j≠i of A[i, j]. The same margin under shuffled bridges is reported as a reference.
- **Across transitions:** Spearman ρ between margin_i and test gain_i (mean `joint_rr` bridge − no_change). With five transitions this is descriptive. The exact permutation p is reported, but its smallest possible value is 1/60.

**Why a correlation is not enough.** Any direction closer to the visual direction retrieves better, regardless of where it came from. In the synthetic world where text carries **no** information, alignment still correlated with gain (ρ = +1.0 across transitions; r = +0.57, p = 0.0002 across object-subset directions). A positive alignment→gain correlation therefore says nothing about language by itself. The within-transition object-subset correlation originally drafted here was dropped for this reason, before any real data was run.

**Calibration (what replaces it).** For each transition, random unit directions are built at fixed cosines {0, .2, .4, .6, .8, .9, .95, 1} to the training visual direction, with 20 draws per level, and applied at the bridge's λ. This gives the test gain expected from alignment alone, with a 90% band.

- A bridge gain inside the band: the bridge's retrieval is fully explained by its alignment. M1's alignment matrix then converts directly into an expected retrieval gain.
- A bridge gain above the band: the language direction carries held-out-family information beyond its agreement with the training visual direction.

Whether language encodes the state change is answered by the bridge's alignment against the shuffled-bridge alignment (margin vs. shuffled 95th percentile) and by the decision rule above, not by this correlation.

## Claims this design can and cannot support

- A bridge > no_change **and** bridge > shuffled_bridge **and** bridge > other_transition result, with object retention reported, supports "the language direction carries transferable, state-specific visual information through this bridge, for these families".
- Endpoint mapping alone can produce difference transfer (sz684 report §7). That is why the direction is estimated from *training* objects and evaluated on *held-out families*, and why `other_transition` is a control.
- A null result is reported as such. It is a negative result for this bridge, these models and this split, not for PRH in general.
- The MIT-States labels are noisy. Per-transition query counts are reported, and transitions with fewer than 10 test queries are flagged.

## Post-freeze changes

(none yet)
