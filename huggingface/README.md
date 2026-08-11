---
pretty_name: ChessFraud
license: cc-by-4.0
task_categories:
- tabular-classification
tags:
- chess
- cheating-detection
- human-aligned-models
- behavioral-modeling
size_categories:
- 1M<n<10M
configs:
- config_name: chess_fraud
  default: true
  data_files:
  - split: full
    path: data/chess_fraud/full.parquet
- config_name: chess_fraud_synth
  data_files:
  - split: train
    path: data/chess_fraud_synth/train.parquet
  - split: test
    path: data/chess_fraud_synth/test.parquet
---

# ChessFraud

ChessFraud is a tabular benchmark for cheating detection in online chess. It
accompanies the KDD 2026 Datasets and Benchmarks Track paper *ChessFraud:
Exploring the Capabilities of Human-Aligned Models for Cheating Detection in
Online Chess*.

The two configurations serve complementary roles. **ChessFraud** supports
evaluation against cheating observed under a controlled assistance protocol.
**ChessFraud-Synth** supports training and analysis using alternatives produced
by classical engines and human-aligned chess models for public Lichess games.

## Dataset at a glance

| Configuration | Source | Released rows | Key analysis counts |
|---|---|---:|---|
| ChessFraud | 505 controlled games, represented as 1,010 player-side games | 38,510 | 407 cheating player-games (40.3%); 9,405 cheating half-moves (24.4%) |
| ChessFraud-Synth | 12,000 public Lichess player-games | 1,074,287 | 417,207 decision points marked by `is_used` |

## Loading the data

```python
from datasets import load_dataset

chess_fraud = load_dataset(
    "artemlepin/chess-fraud",
    "chess_fraud",
)

chess_fraud_synth = load_dataset(
    "artemlepin/chess-fraud",
    "chess_fraud_synth",
)
```

## Allie move embeddings

The additive `artifacts/allie_embeddings/` subtree contains frozen 1024-dimensional
`float16` move embeddings produced with Allie-2500 for observed moves and for
moves proposed by Stockfish, Lc0, Maia-2, and Allie. It does not change either
tabular configuration or its Parquet schema. Allie uses `elo_oppo` as the
opponent's rating. Downstream experiments select rows with `is_used = true`.

```text
artifacts/allie_embeddings/
├── manifest.json
├── synth/
│   ├── row_index.parquet
│   ├── move_uci.npy
│   ├── move_stockfish_1.npy
│   ├── move_stockfish_5.npy
│   ├── move_stockfish_9.npy
│   ├── move_stockfish_11.npy
│   ├── move_stockfish_15.npy
│   ├── move_lc0_1.npy
│   ├── move_lc0_10.npy
│   ├── move_lc0_100.npy
│   ├── move_maia2_2050.npy
│   └── move_allie_2500.npy
└── tournament/
    ├── row_index.parquet
    └── embs_allie_2500.npz
```

The eleven Synth arrays preserve their source `.npy` bytes, row order, shape,
and dtype. The tournament `.npz` likewise preserves its source bytes and
contains the named `move_uci` array. These embeddings are used by the
move-level and game-level experiments reported in Tables 4 and 5. The matching
GitHub notebooks provide the reproduction protocol and loading details.

### Alignment indexes

Join each `row_index.parquet` to the matching tabular configuration using the
documented keys, then use `embedding_row` to index the arrays. Never assume
Parquet row order. Each index provides a zero-based, contiguous, unique
`embedding_row` and a unique key for a one-to-one join.

| Index | Columns in schema order | Join key |
|---|---|---|
| `synth/row_index.parquet` | `embedding_row` (`int64`), `player_id` (`string`), `game_id` (`string`), `half_move` (`int64`), `move_player` (`string`), `move_thinking_time` (`float64`) | `player_id`, `game_id`, `half_move`, `move_player` |
| `tournament/row_index.parquet` | `embedding_row` (`int64`), `game_id` (`string`), `player_id` (`int64`), `half_move` (`int64`), `move_player` (`string`) | `game_id`, `player_id`, `half_move`, `move_player` |

The Synth index retains the finite source-manifest `move_thinking_time` needed
by the embedding experiments. This value is authoritative when an impossible
source clock interval is intentionally represented as missing in the public
core table.

### Provenance and integrity

`artifacts/allie_embeddings/manifest.json` defines the `allie-embeddings-v1` contract. It records
the complete file inventory, byte sizes, SHA-256 checksums, NumPy shapes and
dtypes, the tournament archive member metadata, index schemas and row counts,
normalized row digests, source provenance, and the path-neutral preparation
command. Validate the manifest and the one-to-one joins before training; a
checksum match alone does not establish row alignment.

### Download, cache, and local override

The embedding subtree is approximately 9.56 GB, excluding Hugging Face cache and
temporary-file overhead. Ensure that both download storage and working space
are available before running a matching tutorial. The tutorials pin a full
immutable 40-character dataset revision; use exactly the revision recorded by
the tutorial you are reproducing, rather than a mutable branch or tag.

The following helper downloads only the embedding subtree. Its `revision`
argument must be the full immutable revision copied from the matching GitHub
tutorial.

```python
from huggingface_hub import snapshot_download


def download_allie_embeddings(*, revision: str, cache_dir: str | None = None) -> str:
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision.lower()
    ):
        raise ValueError("revision must be a full 40-character hexadecimal commit SHA")
    return snapshot_download(
        repo_id="artemlepin/chess-fraud",
        repo_type="dataset",
        revision=revision,
        allow_patterns=["artifacts/allie_embeddings/**"],
        cache_dir=cache_dir,
    )
```

The matching repository tutorials provide a validated resolver with two
environment overrides:

- `CHESSFRAUD_ALLIE_CACHE_DIR` selects the snapshot cache directory;
- `CHESSFRAUD_ALLIE_EMBEDDING_DIR` bypasses downloading and points to an
  existing snapshot root that already contains a validated
  `artifacts/allie_embeddings/` subtree.

The local override takes precedence over downloading, but it does not bypass
manifest, checksum, schema, or row-index validation.

### Scope, privacy, licensing, and limitations

This subtree contains derived embeddings and alignment metadata for rows that
are already represented in the public benchmark. The indexes introduce no new
identifier domain: tournament identifiers remain pseudonymous, whereas Synth
identifiers remain the public Lichess usernames and game IDs described below.
The bundle excludes personal information, raw server logs, private engine-hint
traces, the tournament plugin, model weights, and detector outputs.

The dataset-level licensing and source terms in this card also apply to the
embedding bundle: the ChessFraud release is CC BY 4.0, ChessFraud-Synth derives
from the public Lichess database under CC0, and model weights and engine
binaries remain under their original licenses. The embeddings cover only the
frozen representations and engine configurations used by the embedding
experiments; they are not a general model release, do not expand the
annotation scope, and should not be used as evidence for sanctions against
individual players.

## Data representation

Both configurations are organized around a focal player, identified by
`player_id`.

In ChessFraud, each physical half-move is stored once, in the player-side game
of the player who made the move. `game_id` retains the source game hash and has
the suffix `_w` or `_b` for the White or Black player-side view.

In ChessFraud-Synth, `player_id` remains fixed within a player-game and the full
sequence is retained. Rows for both players provide preceding game context.

All positions are encoded in FEN. All moves are encoded in UCI.

## Shared fields

Unless stated otherwise, every field in this section is present in both
configurations. All evaluations and outcome estimates use the perspective of
`player_id`: a positive evaluation favors the focal player, irrespective of
color.

### Identifiers

| Field | Type | Values | Meaning |
|---|---|---|---|
| `game_id` | string | non-empty | Player-side game identifier. ChessFraud uses a pseudonymous game hash with a color suffix; ChessFraud-Synth retains the public Lichess game ID. |
| `player_id` | int64 in ChessFraud; string in ChessFraud-Synth | ChessFraud: `1`–`49`; ChessFraud-Synth: non-empty | Focal player. ChessFraud uses a pseudonymous integer; ChessFraud-Synth retains the public Lichess username. |
| `opponent_id` | int64 in ChessFraud; string in ChessFraud-Synth | same identifier convention as `player_id` | Opponent of `player_id`. |

### Player context

| Field | Type | Values | Meaning |
|---|---|---|---|
| `player_color` | string | `white`, `black` | Color of `player_id`. |
| `player_elo` | int64 | non-negative | Rating of `player_id` for the game. |
| `opponent_elo` | int64 | non-negative | Rating of `opponent_id` for the game. |
| `time_control` | string | `base+increment` | Base time and increment in seconds. ChessFraud uses `300+0`. |
| `game_result` | string | `1-0`, `0-1`, `0.5-0.5` | Game result from White's perspective. |

ChessFraud ratings are standardized to the Lichess scale. A reported Lichess
rating is used when available. For four participants who reported only a
Chess.com rating, 300 points are added to approximate the corresponding
Lichess rating. This procedure is described in
[Appendix C.1 of the paper](https://doi.org/10.1145/3770855.3817587).

### Move context

| Field | Type | Values | Meaning |
|---|---|---|---|
| `half_move` | int64 | `>= 1` | One-based half-move index within the source game. |
| `move_player` | string | legal UCI move | Observed move at the current half-move. On ChessFraud-Synth analysis rows, it is the focal player's move. |
| `position_before` | string | valid FEN | Position immediately before `move_player`. |
| `position_after` | string | valid FEN | Position immediately after `move_player`. |

### Timing

Timing values are measured in seconds. Their units are stated here rather than
repeated in the field names.

In ChessFraud-Synth, impossible negative intervals in the public Lichess clock
telemetry are stored as missing `move_thinking_time` values.

| Field | Type | Range | Meaning |
|---|---|---|---|
| `move_thinking_time` | float64 | `[0, +inf)` | Time spent on the observed move. |
| `clock_remaining_time` | float64 | `[0, +inf)` | Clock time remaining for the player who made the observed move. |

### Analysis selection

| Field | Type | Values | Meaning |
|---|---|---|---|
| `is_used` | bool | `true`, `false` | Whether the row is a paper-aligned decision point. |

The opening cutoff is 20 half-moves. In ChessFraud, `is_used` is true for a
retained player move with `half_move >= 21`. In ChessFraud-Synth, it is true
when `half_move >= 21` and the side to move matches `player_color`: odd
half-moves for White and even half-moves for Black.

### Evaluation features

`eval_before`, `eval_after`, and every `eval_<source>` field are Stockfish 17
depth-15 evaluations. They are expressed in centipawns from the perspective of
`player_id` and clipped to `[-1000, 1000]`, corresponding to `[-10, 10]` pawns.

| Field | Type | Range | Meaning |
|---|---|---|---|
| `eval_before` | int64 | `[-1000, 1000]` | Clipped evaluation of `position_before`. |
| `eval_after` | int64 | `[-1000, 1000]` | Clipped evaluation after `move_player`, reoriented to the same player. |
| `centipawn_loss` | int64 | `[0, 2000]` | Loss in clipped evaluation after the observed move: `max(0, eval_before - eval_after)`. |
| `normalized_centipawn_loss` | float64 | `[0, 1]` | Decrease in the current player's estimated winning chance after the observed move, computed using the [Lichess Win% model](https://lichess.org/page/accuracy). |

Normalized centipawn loss converts the published clipped evaluations using

```text
p(e) = 1 / (1 + exp(-k * e)), where k = 0.00368208
normalized_centipawn_loss = max(0, p(eval_before) - p(eval_after))
```

### Engine outputs

The `move_<source>` fields in this and the following section are post hoc
features in ChessFraud and candidate synthetic assistance moves in
ChessFraud-Synth. All published Stockfish feature outputs were computed with
Stockfish 17 and are distinct from the Stockfish 15 assistance recorded during
the controlled tournaments.

| Field | Type | Values | Meaning |
|---|---|---|---|
| `move_stockfish_1` | string | UCI move | Top-1 move at depth 1. |
| `move_stockfish_9` | string | UCI move | Top-1 move at depth 9. |
| `move_stockfish_15` | string | UCI move | Top-1 move at depth 15. |

### Human-aligned model outputs

Maia-2 uses a focal-player rating of 2050, while Allie uses 2500. Both models
receive the opponent's recorded rating for the game. Allie additionally uses
the time control and preceding move history available at the current position.

| Field | Type | Range or values | Meaning |
|---|---|---|---|
| `move_maia2_2050` | string | UCI move | Maia-2 top-1 move. |
| `maia2_win_prob_2050` | float64 | `[0, 1]` | Player-perspective game-outcome estimate: loss = 0, draw = 0.5, win = 1. |
| `move_allie_2500` | string | UCI move | Allie top-1 move. |
| `allie_win_prob_2500` | float64 | `[0, 1]` | Player-perspective game-outcome estimate: loss = 0, draw = 0.5, win = 1. |

Engine and human-aligned model outputs are filled for every row where `is_used`
is true. Other rows are retained only as sequence context, so these outputs can
be empty there.

## ChessFraud fields

The fields in this section appear only in the `chess_fraud` configuration.
ChessFraud was collected in two controlled tournaments, numbered 4 and 5. The
tournament assistance plugin used Stockfish 15, and each participant was
assigned a search depth of 12, 14, 16, or 18.

| Field | Type | Range or values | Meaning |
|---|---|---|---|
| `tournament_id` | int64 | `4`, `5` | Controlled tournament. |
| `player_hint_shown` | bool | `true`, `false` | Whether a Stockfish hint was shown to the current player on this half-move. |
| `assistance_search_depth` | int64 | `12`, `14`, `16`, `18` | Search depth assigned to the player in the tournament plugin. |
| `assistance_line_rank` | int64 | `1`–`5` | One-based rank of the displayed engine line followed by the player. |
| `is_cheating_move` | bool | `true`, `false` | Whether the move belongs to a followed Stockfish line under the tournament annotation protocol. |
| `is_cheating_player_game` | bool | `true`, `false` | Player-game cheating label used for game-level evaluation. |
| `is_accused_by_opponent` | bool | `true`, `false` | Whether the opponent accused `player_id` of cheating after the game. |

A cheating series starts when a player follows a displayed Stockfish line and
continues while the player follows the same continuation. The series ends when
the player diverges from that line. A hint could be requested during the
opponent's turn; in that case, the next matching player move is labeled even
though `player_hint_shown` is false on that move's row.

`assistance_line_rank` is present only when the player followed one of the
displayed engine lines.

## ChessFraud-Synth fields

The fields in this section appear only in the `chess_fraud_synth`
configuration. They extend the shared fields with the remaining top-1 outputs
used in the paper. Stockfish and Lc0 are classical engines; Maia-1, Maia-2, and
Allie are human-aligned models.

### Sampling metadata

| Field | Type | Values | Meaning |
|---|---|---|---|
| `rating_bin` | string | six 200-point Elo strata up to 2200 | Rating stratum used to sample the focal player. |

### Additional engine outputs

| Field | Type | Range or values | Meaning |
|---|---|---|---|
| `move_stockfish_5` | string | UCI move | Top-1 move at depth 5. |
| `move_stockfish_11` | string | UCI move | Top-1 move at depth 11. |
| `eval_stockfish_1` | int64 | `[-1000, 1000]` | Evaluation after `move_stockfish_1`. |
| `eval_stockfish_5` | int64 | `[-1000, 1000]` | Evaluation after `move_stockfish_5`. |
| `eval_stockfish_9` | int64 | `[-1000, 1000]` | Evaluation after `move_stockfish_9`. |
| `eval_stockfish_11` | int64 | `[-1000, 1000]` | Evaluation after `move_stockfish_11`. |
| `eval_stockfish_15` | int64 | `[-1000, 1000]` | Evaluation after `move_stockfish_15`. |
| `move_lc0_1` | string | UCI move | Lc0 top-1 move with a 1-node budget. |
| `move_lc0_10` | string | UCI move | Lc0 top-1 move with a 10-node budget. |
| `move_lc0_100` | string | UCI move | Lc0 top-1 move with a 100-node budget. |
| `eval_lc0_1` | int64 | `[-1000, 1000]` | Evaluation after `move_lc0_1`. |
| `eval_lc0_10` | int64 | `[-1000, 1000]` | Evaluation after `move_lc0_10`. |
| `eval_lc0_100` | int64 | `[-1000, 1000]` | Evaluation after `move_lc0_100`. |

### Additional human-aligned model outputs

| Field | Type | Range or values | Meaning |
|---|---|---|---|
| `move_maia1_1900` | string | UCI move | Maia-1 1900 top-1 move. |
| `eval_maia1_1900` | int64 | `[-1000, 1000]` | Evaluation after `move_maia1_1900`. |
| `eval_maia2_2050` | int64 | `[-1000, 1000]` | Evaluation after `move_maia2_2050`. |
| `eval_allie_2500` | int64 | `[-1000, 1000]` | Evaluation after `move_allie_2500`. |

The `eval_<source>` fields allow source strength to be computed under one
evaluation protocol. For a proposed move, its centipawn loss is
`max(0, eval_before - eval_<source>)`. No duplicate centipawn-loss columns are
stored.

## Splits

ChessFraud has one `full` split. ChessFraud-Synth uses the paper's 80/20
player-disjoint `train` and `test` split, built from the source
`split_by_player` field. The source `split_by_player` and `split_by_games`
columns are not included in the released tables. ChessFraud-Synth and
ChessFraud do not share players.

## Reconstructing synthetic examples

Each row stores the observed move once. A binary fair/assisted example for a
chosen source can be constructed without generating new chess features:

- use `move_player` as the fair move;
- use one `move_<source>` field as the assisted move;
- assign labels 0 and 1, respectively;
- restrict paper-aligned experiments to rows where `is_used` is true.

The release does not store a generic `move_assistance` or `assistance_model`
column because all published source moves are explicit columns.

## Dataset construction

### ChessFraud

The controlled collection and annotation protocol is described in
[Section 3 and Appendix C of the paper](https://doi.org/10.1145/3770855.3817587).
Appendix C.1 documents rating standardization, and Appendix C.4 gives the engine
usage instructions provided to participants.

### ChessFraud-Synth

ChessFraud-Synth was constructed from public Lichess Blitz games played in
March 2025 using rating-stratified sampling. Matches between two selected
players were excluded from the sampled data.

The sampling design, temporal separation from the human-aligned models'
training data, split construction, and synthetic cheating procedure are
described in
[Section 4.3 of the paper](https://doi.org/10.1145/3770855.3817587).

## Limitations

1. ChessFraud covers a small controlled cohort playing five-minute games and
   does not represent every player population or time control.
2. ChessFraud-Synth contains model-generated move substitutions rather than
   observed cheating behavior.
3. ChessFraud labels cover the logged Stockfish assistance protocol; other
   forms of assistance are outside the annotation scope.
4. The tournament cheating plugin is not released.

## Privacy, consent, and responsible use

All ChessFraud participants gave informed consent to share their gameplay data
under the controlled protocol. Compensation was fixed and independent of
tournament performance. The release uses pseudonymous participant identifiers;
personal information and raw server logs are excluded.

ChessFraud-Synth retains public Lichess usernames and game IDs so that source
games remain auditable. These identifiers come from the public Lichess game
database.

The datasets are intended for research on cheating detection. They are not a
basis for sanctions against individual players in unrelated games.

## Licensing and source terms

The ChessFraud release is licensed under CC BY 4.0. ChessFraud-Synth is derived
from games distributed through the public Lichess database under CC0. Model
weights and engine binaries remain subject to their original licenses.

## Citation

If you use ChessFraud or ChessFraud-Synth, cite the accepted paper:

```bibtex
@inproceedings{linich2026chess_fraud,
  title     = {ChessFraud: Exploring the Capabilities of Human-Aligned Models for Cheating Detection in Online Chess},
  author    = {Linich, Anastasiia and Lepin, Artem and Sakhovskiy, Andrey and Toleutaeva, Anita and Lepa, Georgii and Neznamov, Andrei and Budennyy, Semen},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year      = {2026},
  doi       = {10.1145/3770855.3817587}
}
```

Paper: <https://doi.org/10.1145/3770855.3817587>
