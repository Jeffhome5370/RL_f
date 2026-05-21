import random

from config import ACTIONS, SEED
from data.dataset_manager import DatasetManager
from env.tool_selection_env import ToolSelectionEnv
from utils.seed import set_seed


def main():
    set_seed(SEED)

    dataset_manager = DatasetManager(seed=SEED)
    datasets = dataset_manager.load_all()

    env = ToolSelectionEnv(datasets=datasets, seed=SEED)

    state = env.reset()

    print("Initial state:")
    print(state)

    done = False
    total_reward = 0.0

    while not done:
        action_id = random.randint(0, len(ACTIONS) - 1)
        action = ACTIONS[action_id]

        next_state, reward, done, info = env.step(action_id)

        print("=" * 50)
        print(f"Action: {action}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Info: {info}")

        total_reward += reward
        state = next_state

    print("=" * 50)
    print(f"Total reward: {total_reward}")
    print(f"Final pipeline: {env.pipeline_actions}")
    print(f"Invalid count: {env.invalid_count}")

if __name__ == "__main__":
    main()
