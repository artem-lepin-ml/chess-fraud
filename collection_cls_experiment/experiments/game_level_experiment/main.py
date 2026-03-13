import pandas as pd
from pathlib import Path

from collection_method import get_collection_method_results
from baselines import human_accusation, st_top_match, always_cheat
from consts import TOURNAMENT_DF


def main():
    results = get_collection_method_results()
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    results.to_csv(results_dir / "collection_method_results.csv")
    
    baselines_results = []
    tournament = pd.read_csv(TOURNAMENT_DF)
    baselines_results.append(st_top_match(tournament))
    baselines_results.append(human_accusation(tournament))
    baselines_results.append(always_cheat(tournament))
    baselines_results.to_csv(results_dir / "baselines.csv")


if __name__ == "__main__":
    main()