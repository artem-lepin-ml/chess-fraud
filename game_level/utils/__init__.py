# Utils package containing shared code for collection classification

from .common_utils import seed_everything
from .config_utils import (
    load_config,
    load_embeddings,
    load_pickle,
    save_pickle,
    ModelConfig,
    TrainingConfig,
    CollectionConfig,
    AllCheatColumns,
)
from .collection_utils import (
    # Data classes
    CollectionIndices,
    CollectionSample,
    DatasetIndicesData,
    # Legacy (2-class) functions
    choose_cheat_column,
    make_cheat_move_index,
    make_fair_move_index,
    get_one_collection_embs_indices,
    make_player_collections_indices,
    get_one_collection_embs_from_indices,
    make_player_collections_from_indices,
    make_collections_indices_for_all_players,
    make_tournament_collections_from_indices,
    make_collections_from_indices,
    # New (3-class) functions
    get_one_collection_embs_indices_3cls,
    make_player_collections_indices_3cls,
    make_collections_indices_for_all_players_3cls,
    get_one_collection_embs_from_indices_3cls,
    make_player_collections_from_indices_3cls,
    make_collections_from_indices_3cls,
)

__all__ = [
    "seed_everything",
    "load_config",
    "load_embeddings",
    "load_pickle",
    "save_pickle",
    "ModelConfig",
    "TrainingConfig",
    "CollectionConfig",
    "AllCheatColumns",
    "CollectionIndices",
    "CollectionSample",
    "DatasetIndicesData",
    "choose_cheat_column",
    "make_cheat_move_index",
    "make_fair_move_index",
    "get_one_collection_embs_indices",
    "make_player_collections_indices",
    "get_one_collection_embs_from_indices",
    "make_player_collections_from_indices",
    "make_collections_indices_for_all_players",
    "make_tournament_collections_from_indices",
    "make_collections_from_indices",
    "get_one_collection_embs_indices_3cls",
    "make_player_collections_indices_3cls",
    "make_collections_indices_for_all_players_3cls",
    "get_one_collection_embs_from_indices_3cls",
    "make_player_collections_from_indices_3cls",
    "make_collections_from_indices_3cls",
]
