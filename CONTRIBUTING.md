# Contributing to SolarScan

The course grades commit history: every member must commit **from their own GitHub
account**, throughout the project. A repository uploaded in one commit loses the
code-quality marks, and a member with no meaningful commits can lose up to half of the team
mark.

## Workflow

1. Pull the latest `main`.
   ```bash
   git checkout main && git pull
   ```
2. Create a feature branch named `<area>/<short-description>`:
   ```bash
   git checkout -b enhancement/histogram-equalization
   ```
3. Commit small, meaningful steps. Message format: `<area>: <what changed>`, e.g.
   `enhancement: add NumPy histogram equalization`.
4. Run the checks before pushing:
   ```bash
   flake8 src tests
   pytest
   ```
5. Push and open a pull request into `main`.
6. **A different member reviews and approves** the PR before it is merged. Never merge your
   own PR.
7. Squash or merge, then delete the branch.

## Code rules (from the course spec)

- Every function has a docstring: what it does, parameters with dtype and range, return value.
- Lines of at most **95 characters** (enforced by `flake8`, see `setup.cfg`).
- **No hard-coded paths.** Paths come from configs or command-line arguments.
- **No warnings** in the output.
- Images are loaded only through `src.data.load_image`. No module reloads images its own way.
- Convert BGR → RGB before displaying any image.
- Every NumPy implementation has a matching test in `tests/` that:
  - checks it against the OpenCV / scikit-image function (`np.array_equal` or max abs diff),
  - checks a 3×3 or 5×5 hand-worked example.
- State border handling (zero / replicate / reflect) in the docstring of every neighbourhood
  operation, and use the same mode in the NumPy and library versions.

## Experiments

Every run is logged as one row in `experiments/experiment_log.csv`
(run_id, date, author, pipeline, stage, method, params, seed, time_s, metric_name,
metric_value_dev). Tune on the **development set only**. The test split is never opened before
Week 13.

## Data

Never commit raw images or annotations. Put them under `data/raw/` (ignored by git).

## AI assistants

Allowed as a help, not as an author. You are examined on every line in the repository. Note
where you used an AI tool in your PR description so it can be reported in the final report.
