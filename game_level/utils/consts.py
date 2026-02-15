"""
Constants module that provides backward compatibility.

Loads values from config.yaml and makes them available as module-level constants.
"""

from .config_utils import AllCheatColumns, CollectionConfig, load_config
import os

# Load config
config_paths = [
    "config.yaml",
    "../config.yaml",
    "../../config.yaml",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml"),
]
_config = None
for path in config_paths:
    if os.path.exists(path):
        try:
            _config = load_config(path)
            break
        except Exception:
            continue

if _config is None:
    raise FileNotFoundError("Could not find config.yaml")

_all_cheat = AllCheatColumns.from_cfg(_config)
_collection_cfg = CollectionConfig.from_cfg(_config)

# Export constants - loaded from config.yaml
ALLIE_CHEAT_COLUMNS = _all_cheat.allie
MAIA2_CHEAT_COLUMNS = _all_cheat.maia2
K_MIN = _collection_cfg.k_min
K_MAX = _collection_cfg.k_max
MIN_HMOVE = _collection_cfg.min_hmove
MAX_HMOVE = _collection_cfg.max_hmove
MIN_CHEAT_P = _collection_cfg.min_cheat_p
MAX_CHEAT_P = _collection_cfg.max_cheat_p
