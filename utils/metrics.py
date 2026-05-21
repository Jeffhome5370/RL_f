import numpy as np
import pandas as pd


def summarize_dqn_results(results):
    rewards = [r["reward"] for r in results]
    f1s = [r["f1"] for r in results if r["f1"] is not None]
    invalids = [r["invalid_count"] for r in results]
    lengths = [r["pipeline_length"] for r in results]

    summary = {
        "avg_reward": float(np.mean(rewards)),
        "avg_f1": float(np.mean(f1s)) if len(f1s) > 0 else None,
        "avg_invalid_count": float(np.mean(invalids)),
        "avg_pipeline_length": float(np.mean(lengths)),
    }

    return summary


def save_results_csv(results, path):
    df = pd.DataFrame(results)
    df.to_csv(path, index=False, encoding="utf-8-sig")
