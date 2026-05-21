import os

import pandas as pd

from baselines.fixed_pipeline import run_fixed_baselines
from baselines.grid_search_baseline import run_grid_search_baseline
from baselines.random_agent import run_random_agent
from config import SEED, TABLE_DIR
from data.dataset_manager import DatasetManager
from env.tool_selection_env import ToolSelectionEnv
from utils.seed import set_seed


def run_all_baselines():
    set_seed(SEED)

    os.makedirs(TABLE_DIR, exist_ok=True)

    dataset_manager = DatasetManager(seed=SEED)
    datasets = dataset_manager.load_all()

    env = ToolSelectionEnv(datasets=datasets, seed=SEED)

    random_results = run_random_agent(env, episodes=100)
    pd.DataFrame(random_results).to_csv(
        os.path.join(TABLE_DIR, "random_agent_results.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    fixed_results = run_fixed_baselines(datasets, seed=SEED)

    fixed_rows = []
    for method_name, rows in fixed_results.items():
        for r in rows:
            r["method"] = method_name
            fixed_rows.append(r)

    pd.DataFrame(fixed_rows).to_csv(
        os.path.join(TABLE_DIR, "fixed_pipeline_results.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    grid_results = run_grid_search_baseline(datasets, seed=SEED)
    pd.DataFrame(grid_results).to_csv(
        os.path.join(TABLE_DIR, "grid_search_results.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("Baseline experiments finished.")
    print(f"Tables saved to {TABLE_DIR}")


if __name__ == "__main__":
    run_all_baselines()
