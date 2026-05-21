import os

import numpy as np
from tqdm import tqdm

from agents.dqn_agent import DQNAgent
from config import (
    ACTION_DIM,
    BATCH_SIZE,
    EPS_DECAY,
    EPS_END,
    EPS_START,
    EPISODES,
    FIGURE_DIR,
    GAMMA,
    LOG_DIR,
    LR,
    MIN_REPLAY_SIZE,
    REPLAY_BUFFER_SIZE,
    SEED,
    STATE_DIM,
    TARGET_UPDATE_FREQ,
)
from data.dataset_manager import DatasetManager
from env.tool_selection_env import ToolSelectionEnv
from utils.metrics import save_results_csv
from utils.plot import moving_average, plot_curve
from utils.seed import set_seed


def train_dqn():
    set_seed(SEED)

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    dataset_manager = DatasetManager(seed=SEED)
    datasets = dataset_manager.load_all()

    env = ToolSelectionEnv(datasets=datasets, seed=SEED)

    agent = DQNAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        lr=LR,
        gamma=GAMMA,
        buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
    )

    epsilon = EPS_START

    episode_rewards = []
    episode_losses = []
    episode_f1s = []
    episode_invalids = []
    episode_lengths = []

    results = []

    for ep in tqdm(range(EPISODES), desc="Training DQN"):
        state = env.reset()
        done = False

        total_reward = 0.0
        losses = []
        final_info = None

        while not done:
            action_id = agent.select_action(state, epsilon)
            next_state, reward, done, info = env.step(action_id)

            agent.replay_buffer.push(state, action_id, reward, next_state, done)

            state = next_state
            total_reward += reward
            final_info = info

            if len(agent.replay_buffer) >= MIN_REPLAY_SIZE:
                loss = agent.update()
                if loss is not None:
                    losses.append(loss)

        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        if ep % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        avg_loss = np.mean(losses) if len(losses) > 0 else None

        episode_rewards.append(total_reward)
        episode_losses.append(avg_loss if avg_loss is not None else 0)
        episode_f1s.append(final_info.get("f1") if final_info.get("f1") is not None else 0)
        episode_invalids.append(env.invalid_count)
        episode_lengths.append(len(env.pipeline_actions))

        results.append(
            {
                "episode": ep,
                "dataset": final_info.get("dataset"),
                "reward": total_reward,
                "loss": avg_loss,
                "epsilon": epsilon,
                "f1": final_info.get("f1"),
                "pipeline": final_info.get("pipeline"),
                "invalid_count": env.invalid_count,
                "pipeline_length": len(env.pipeline_actions),
                "cache_hit": final_info.get("cache_hit"),
            }
        )

    save_results_csv(results, os.path.join(LOG_DIR, "dqn_training_results.csv"))

    plot_curve(
        moving_average(episode_rewards),
        "DQN Reward Curve",
        "Reward",
        os.path.join(FIGURE_DIR, "dqn_reward_curve.png"),
    )

    plot_curve(
        moving_average(episode_f1s),
        "DQN F1 Curve",
        "F1 Score",
        os.path.join(FIGURE_DIR, "dqn_f1_curve.png"),
    )

    plot_curve(
        moving_average(episode_invalids),
        "DQN Invalid Action Curve",
        "Invalid Actions",
        os.path.join(FIGURE_DIR, "dqn_invalid_curve.png"),
    )

    plot_curve(
        moving_average(episode_lengths),
        "DQN Pipeline Length Curve",
        "Pipeline Length",
        os.path.join(FIGURE_DIR, "dqn_pipeline_length_curve.png"),
    )

    agent.save(os.path.join(LOG_DIR, "dqn_agent.pth"))

    print("Training finished.")
    print(f"Results saved to {LOG_DIR}")
    print(f"Figures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    train_dqn()
