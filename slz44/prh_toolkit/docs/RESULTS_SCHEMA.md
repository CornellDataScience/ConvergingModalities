# Results format, version 0.1

One JSON file per run. Every experiment writes one, and the toolkit and visualizer read them.
Create with `prh.new_run`, add rows with `prh.add_record`, save with `prh.write`, check with `prh validate`.

## Run (top level)

| Field | Meaning |
| --- | --- |
| `schema_version` | `"0.1"` |
| `run_id` | 3-64 chars, lowercase letters, digits, `.` `_` `-` |
| `track` | `"1"`, `"2"` or `"shared"` |
| `created` | ISO 8601 timestamp |
| `git_commit` | short or full hash, or `"uncommitted"` |
| `protocol` | `{file, sha256}`: the plan this run followed, committed before the run |
| `dataset` | `{name, n_pool?}` |
| `models` | list of `{id, modality}`; modality is `language`, `vision`, `audio` or `other` |
| `records` | list of records (below) |
| `notes` | optional free text |

## Record (one measured number)

Required: `metric` (string), `condition` (string, for example `center` or `shared_text`), `value` (finite number).

Optional: `n`, `k`, `fold`, `seed` (integers); `split`, `relation`, `noun`, `language_model`, `vision_model`
(strings); `null_mean`, `null_q95`, `interval_low`, `interval_high` (numbers); `layers` (names to integers);
`extra` (any object). Unknown top-level record fields are rejected so typos are caught; use `extra`.

## Metric names in use

| Metric | Source |
| --- | --- |
| `knn_excess_fixed_k`, `knn_excess_scaled_k`, `cka_excess`, `gw_cost_reduction` | metric by control by N grid |
| `state_direction_cos_own`, `state_direction_margin`, `state_direction_top1` | Track 1 per-category alignment |

Add new names freely and list them here. Join keys across analyses: `relation` + `noun`, `fold`, `condition`.

## Example

```python
import prh
run = prh.new_run('scale-grid-1', 'shared', 'SCALE_GRID_PROTOCOL.md',
                  dataset={'name': 'WIT', 'n_pool': 1024},
                  models=[{'id': 'BLOOMZ', 'modality': 'language'}, {'id': 'DINOv2', 'modality': 'vision'}])
prh.add_record(run, 'cka_excess', 'center', 0.28, n=512, seed=20261004)
prh.write(run, 'results/scale-grid-1.json')
```
