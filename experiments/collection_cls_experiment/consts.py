BATCH_SIZE = 32
EMB_DIM = 1024
MIN_HMOVE = 21
MAX_HMOVE = 120
# number of samples in the dataset
# from k_min = (40 - 20) / 2 = 10 minimum moves in a game excluding the opening
# to k_max = for example (120 - 20) / 2 = 50 moves
K_MIN = 10
K_MAX = 50

DEVICE = "cuda:0"

N_SPECIAL_TOKENS = 3

MIN_CHEAT_P = 0.1
MAX_CHEAT_P = 1.0


LR = 0.0001
N_EPOCHS = 20


MOVE_LOSS_COEFF = 1.

ALLIE_CHEAT_COLUMNS = [
    'move_maia1_1900',
    'move_lc0_1',
    'move_lc0_100',
    'move_maia2_2050',
    'move_stockfish_1',
    'move_stockfish_15',
    'move_allie_2500',
]
MAIA2_CHEAT_COLUMNS = [
    'fen_maia1_1900',
    'fen_lc0_1',
    'fen_lc0_100',
    'fen_maia2_2050',
    'fen_stockfish_1',
    'fen_stockfish_15',
    'fen_allie_2500'
]

TEST_N_COLLECTIONS = 50
TRAIN_N_COLLECTIONS = 70
TRAIN_INDICES_PATH = "data/train_dataset_indices.pickle"
VAL_INDICES_PATH = "data/val_dataset_indices.pickle"
TEST_INDICES_PATH = "data/test_dataset_indices.pickle"
MAX_HIGH_ELO = 2200
SYNT_DATA_PATH = "YOUR DATA PATH"

ELO_BOUNDS = [1200, 1400, 1600, 1800, 2000, 2200]
OPTIMAL_N_COLS = 50

EMB_DIM = 1024
ENCODER_HIDDEN_DIM = 64
NUM_ATTN_HEADS = 1
NUM_ENCODER_LAYERS = 1
LAST_LAYER_DIM = 16

SYNT_MAIA2_EMBS_PATH = "YOUR DATA PATH"
SYNT_ALLIE_EMBS_PATH = "YOUR DATA PATH"

BIN_WIDTH = 200
MIN_ELO = 1200

PROCESSED_TOURNAMENT_PATH = "YOUR DATA PATH"
TOURNAMENT_MAIA2_EMBS_PATH = "YOUR DATA PATH"
TOURNAMENT_ALLIE_EMBS_PATH = "YOUR DATA PATH"