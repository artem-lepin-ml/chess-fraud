# Chess Fraud Collection Classification Experiment

A machine learning framework for detecting cheating in chess games using collection-based learning and neural network models.

## Overview

This project implements a collection-based approach to chess fraud detection. Instead of analyzing individual moves or games in isolation, the model groups moves into "collections" from each player and learns patterns that distinguish fair play from cheating.

### Key Features

- **Collection-based Learning**: Groups moves from each player into collections for pattern recognition
- **Transformer-based Model**: Uses attention mechanisms to learn move-level and game-level representations
- **Multiple Baselines**: Comprehensive comparison against Stockfish-based matching, constant classification, and Irwin baselines
- **Configurable Experiments**: Easy experimentation with different hyperparameters and settings
- **Tournament-level Evaluation**: Evaluates performance across different ELO ranges and cheating engines

## Project Structure

```
game_level/
├── config.yaml                    # Configuration file (constants organized by meaning)
├── dataset.py                     # PyTorch Dataset class for move collections
├── model.py                       # ChessEncoder neural network model
├── make_collections.py            # Build collections from tournament data
├── main.py                        # Main runner - executes all experiments sequentially
├── utils/
│   ├── __init__.py                # Package exports
│   ├── config_utils.py            # Configuration loading utilities
│   ├── collection_utils.py        # Collection building utilities
│   ├── train_and_eval_utils.py    # Training and evaluation utilities
│   └── consts.py                  # Backward compatibility layer
└── experiments/
    ├── game_level_experiment/     # Game-level classification experiment
    │   ├── irwin.py               # Irwin baseline implementation
    │   ├── collection_method.py   # Collection-based method
    │   ├── baselines.py           # Stockfish & constant baselines
    │   └── main.py
    ├── p_experiment/              # Varying proportion of cheating moves
    │   └── main.py
    └── each_elo_and_cheater_experiment/  # Per-ELO per-cheater analysis
        └── main.py
```

## Data Requirements

The project expects the following data structure:

```
data/
└── processed/
    └── tournament/
        ├── chess_fraud_dataset.csv           # Main tournament data
        └── indices/                          # Collection indices (output of make_collections.py)
    └── embeddings/
        ├── maia2_2050/                        # Maia2 embeddings
        │   └── 04_embs_maia2_2050_full.npz
        └── allie_2500/                        # Allie embeddings
            └── 04_embs_allie_2500_full.npz
```

### Dataset Format (chess_fraud_dataset.csv)

The dataset should contain:
- Tournament metadata (game_id, white_player, black_player, ELOs)
- Move data (fen notation, moves, evaluations)
- Cheating indicators for various engines (maia1_1900, lc0_1, lc0_100, maia2_2050, stockfish_1, stockfish_15, allie_2500)
- Human accusation data

## Installation

### Dependencies

```bash
pip install torch torchvision
pip install omegaconf pyyaml
pip install numpy pandas scikit-learn
pip install matplotlib seaborn
```

### Additional Requirements

- **Irwin**: The `irwin/` folder should be present in `experiments/game_level_experiment/`
- **Chess evaluation engines**: Stockfish, LC0, Maia, Allie (for generating embeddings)

## Configuration

All settings are centralized in `config.yaml`:

### Data Paths
- `csv_path`: Path to main tournament data
- `maia_embeddings_path` / `allie_embeddings_path`: Embedding files
- `data_output_dir`: Where collection indices are saved

### Model Architecture
- `embedding_dim`: 1024
- `encoder_hidden_dim`: 64
- `num_attention_heads`: 1
- `num_encoder_layers`: 1
- `last_layer_dim`: 16

### Training Settings
- `batch_size`: 32
- `learning_rate`: 0.0001
- `n_epochs`: 5
- `move_loss_coeff`: 1.0

### Collection Settings
- `k_min` / `k_max`: Min/max moves per collection
- `min_hmove` / `max_hmove`: Half-move range
- `train_n_collections`: 70 (collections per player for training)
- `test_n_collections`: 50

## Usage

### 1. Build Collections

```bash
python make_collections.py
```

This reads the tournament data and creates collection indices for fair and cheating players.

### 2. Run Single Experiment

Run game-level experiment (all baselines + collection method):
```bash
python experiments/game_level_experiment/main.py
```

Run p-experiment (vary cheating proportions):
```bash
python experiments/p_experiment/main.py
```

Run per-ELO per-cheater experiment:
```bash
python experiments/each_elo_and_cheater_experiment/main.py
```

### 3. Run All Experiments

```bash
python main.py
```

This executes:
1. `make_collections.py` - Build collections from data
2. All three experiments in sequence

Results are saved to `experiments/*/logs/`

## Experiments

### Game Level Experiment

Tests the model's ability to classify games as fair/cheating at the game level.

**Baselines**:
- **Stockfish threshold**: Classifies based on Stockfish first-line match rate
- **Constant class**: Always predicts cheating
- **Irwin**: Irwin's accusation baseline (optimized threshold)

**Collection Method**: Our proposed approach using learned embeddings and attention.

### P Experiment

Studies how performance varies with different proportions of cheating moves (p values from 0.1 to 1.0).

### ELO and Cheater Experiment

Analyzes performance across different ELO ranges (binned by 200 ELO) for each cheating engine.

## Model Architecture

### ChessEncoder

The model uses a transformer encoder with:
- **Input**: Move embeddings (dimension: 1024)
- **Special tokens**: CLS token for game-level classification
- **Encoder**: Multi-head attention layers
- **Outputs**:
  - Move-level predictions (binary: cheat/fair)
  - Game-level predictions (binary: cheat/fair)

### Loss Function

The model optimizes a weighted combination:
```
Loss = move_loss_coeff * move_loss + game_loss
```

## Results

Experiment results are saved as CSV files in `experiments/*/logs/results.csv` containing:
- Accuracy, precision, recall, F1 scores
- Per-ELO breakdowns
- Per-cheating engine performance

## Citation

If you use this code in your research, please cite accordingly.

## License

Please refer to the project license file.
