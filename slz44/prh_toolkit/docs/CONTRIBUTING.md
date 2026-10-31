# Contributing

1. Write the protocol and commit it before running anything. Save its file name and hash in the run.
2. Port your own tested code into the module for your lane (`controls`, `metrics`, `nulls`, `directions`,
   `retrieval`). Do not copy a teammate's code without their review.
3. Every module needs tests that run in `pytest` on small synthetic data.
4. Write results with `prh.schema` and check them with `prh validate`.
5. Work on your own branch and open a pull request into `main`.

## Lane owners (edit as the team agrees)

| Module | Owner |
| --- | --- |
| `schema`, `controls`, `metrics`, `nulls` | M1 |
| `directions` | M1, M2 |
| `retrieval` | M2, M5 |
| visualizer (reads results files) | M3 |
