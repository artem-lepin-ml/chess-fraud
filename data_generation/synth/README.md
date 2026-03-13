# humanlike — Chess “human-aligned cheating detection” experiments

This repo is a multi-stage DS pipeline (Cookiecutter-DS style): **code** lives in `pipelines/`, **run configs** in `configs/`, and **artifacts/data** in `../../data/`.

The key convention is: **stage number matches across folders**:

- `pipelines/<NN>_<stage_name>/...` — executable scripts
- `configs/<NN>_<stage_name>/...` — YAML configs for those scripts
- `../../data/...` — inputs/outputs produced by stages

---

## Repository layout (current)

### Code
- `pipelines/` — pipeline scripts
  - `00_download_db/` — download raw Lichess dumps
  - `01_convert_db/` — convert PGN → CSV/ZST
  - `02_build_dataset/` — select games + annotate moves + build player-move table
  - `03_enrich_dataset/` — add model outputs (Maia2/Stockfish/Maia1/LC0/Allie/characters) + merge tables
  - `04_featurize_dataset/` — compute embeddings / heavy model-derived artifacts
  - `run.sh` — helper runner (optional)
- `utils/` — engine wrappers/helpers (Stockfish, LC0)
  - `utils/stockfish_utils.py`
  - `utils/lc0_utils.py`
- `external_models/maia2/` — vendored Maia2 code (pyproject inside)

### Configs
- `configs/02_build_dataset/`
- `configs/03_enrich_dataset/`
- `configs/04_featurize_dataset/`

**Naming pattern**
`configs/<NN>_<stage>/<SS>_<dataset_id>_<purpose>.yaml`

Where:
- `<NN>_<stage>` — stage folder name (02/03/04)
- `<SS>` — script number inside the stage (`01`, `02`, `03`, …)
- `<dataset_id>` — dataset identifier (see below)
- `<purpose>` — what this run does (`games`, `player_moves`, `maia2`, `stockfish`, `maia2_embs`, …)

Important: **`<SS>` should match the script number** it configures.

### Data
- `data/raw/` — raw inputs
  - large dumps (`*.pgn.zst`, `*.csv.zst`)
  - `example` mini-sample(s)
- `data/external/` — external dependencies (engines, weights)
  - `engines/`, `lc0_weights/`, `maia1_weights/`, `maia2/`, …
- `data/interim/<dataset_id>/` — **main pipeline artifacts per dataset**
  - intermediate CSVs from stages 02–04 (and sometimes helper outputs)
- `data/processed/<dataset_id>/` — “final” consolidated tables used by notebooks/experiments
  - e.g. ready-to-train tables / merged enrichments / embedding bundles

### Notebooks
- `notebooks/experiment_*.ipynb` — main experiment notebooks
- `notebooks/reports/` — outputs (curves, metrics, checkpoints, figures, etc.)

---

## `dataset_id` (what it means)

Every pipeline run materializes into:

`data/interim/<dataset_id>/`

In this repo you currently have:
- `example` — tiny debug dataset
- `13bins_small`
- `13bins_medium`

Rule of thumb:
- Use `example` to validate logic and schemas quickly.
- Use `13bins_*` for real experiments.

---

## Stages (what each one produces)

### Stage 00 — download raw DB
Folder: `pipelines/00_download_db/`

- Downloads Lichess dumps (usually PGN `.zst`).

### Stage 01 — convert raw DB
Folder: `pipelines/01_convert_db/`

- Converts PGN → CSV (often still compressed).
- Outputs typically go to `data/raw/` as `.csv.zst`.

### Stage 02 — build dataset (games → annotated moves → player moves)
Folder: `pipelines/02_build_dataset/`  
Configs: `configs/02_build_dataset/`

Scripts:
- `01_extract_games.py` — select/filter games from raw
- `02_annotate_moves.py` — reconstruct FEN before/after, compute deltas, validate half-moves
- `03_build_player_moves.py` — build player-perspective move table

Typical artifacts end up in `data/interim/<dataset_id>/` and/or `data/processed/<dataset_id>/`
(depending on the script/config).

### Stage 03 — enrich dataset (add engine/model signals)
Folder: `pipelines/03_enrich_dataset/`  
Configs: `configs/03_enrich_dataset/`

Scripts:
- `01_add_maia2.py`
- `02_add_stockfish.py`
- `03_add_maia1.py`
- `04_add_lc0.py`
- `05_add_allie.py`
- `06_add_maia1_characters.py`
- `07_merge_csv_tables.py` — merge side tables back into one wide table (row-aligned)

Outputs are typically “same rows + new columns”, stored under
`data/interim/<dataset_id>/` and/or `data/processed/<dataset_id>/`.

### Stage 04 — featurize dataset (embeddings / heavy artifacts)
Folder: `pipelines/04_featurize_dataset/`  
Configs: `configs/04_featurize_dataset/`

Scripts:
- `01_compute_maia2_embs.py`
- `02_compute_allie_embs.py`
- `03_compute_stockfish_evals.py`

These usually write:
- big arrays (`.npz`, `.npy`) and/or
- feature tables (`.parquet`/`.csv`)
into `data/processed/<dataset_id>/` (or a subfolder next to the table that references them).

---

## Typical usage

All pipeline scripts follow:

```bash
python3 pipelines/<NN>_<stage>/<SS>_<script>.py --config configs/<NN>_<stage>/<SS>_<dataset_id>_<purpose>.yaml
````

Examples (real datasets):

```bash
# Stage 02: build dataset
python3 pipelines/02_build_dataset/01_extract_games.py \
  --config configs/02_build_dataset/01_13bins_small_games.yaml

python3 pipelines/02_build_dataset/03_build_player_moves.py \
  --config configs/02_build_dataset/03_13bins_small_player_moves.yaml


# Stage 03: enrich
python3 pipelines/03_enrich_dataset/01_add_maia2.py \
  --config configs/03_enrich_dataset/01_13bins_small_maia2.yaml

python3 pipelines/03_enrich_dataset/02_add_stockfish.py \
  --config configs/03_enrich_dataset/02_13bins_small_stockfish.yaml

python3 pipelines/03_enrich_dataset/07_merge_csv_tables.py \
  --config configs/03_enrich_dataset/07_13bins_small_merge.yaml


# Stage 04: featurize
python3 pipelines/04_featurize_dataset/01_compute_maia2_embs.py \
  --config configs/04_featurize_dataset/01_13bins_small_maia2_embs.yaml

python3 pipelines/04_featurize_dataset/03_compute_stockfish_evals.py \
  --config configs/04_featurize_dataset/03_13bins_stockfish_eval.yaml
```

For fast debugging, swap `13bins_small` → `example` and use the `*_example_*.yaml` configs.

---

## Where to look when adding a new step

### Add a new enrichment step (Stage 03)

1. Add script to: `pipelines/03_enrich_dataset/` with next number (`08_...py`, etc.)
2. Add config to: `configs/03_enrich_dataset/` with the same prefix (`08_<dataset_id>_<purpose>.yaml`)
3. Write outputs into `data/interim/<dataset_id>/` or `data/processed/<dataset_id>/`
4. Prefer providing an `example` config so it’s runnable on `data/interim/example`

### Add a new featurization step (Stage 04)

Same pattern, but in:

* `pipelines/04_featurize_dataset/`
* `configs/04_featurize_dataset/`

---

## Notes

* `external_models/maia2/` is a vendored dependency; keep it as-is unless you intentionally update the upstream snapshot.
* `data/external/` is for heavyweight dependencies (engines/weights) that aren’t produced by the pipeline.
* `notebooks/` are consumers of `data/processed/<dataset_id>/` and `notebooks/reports/` is where experiment outputs go.