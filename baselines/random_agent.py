import random

from config import ACTIONS


def run_random_agent(env, episodes=100):
    results = []

    for ep in range(episodes):
        state = env.reset()
        done = False

        total_reward = 0.0
        final_info = None

        while not done:
            action_id = random.randint(0, len(ACTIONS) - 1)
            next_state, reward, done, info = env.step(action_id)

            total_reward += reward
            state = next_state
            final_info = info

        results.append(
            {
                "episode": ep,
                "reward": total_reward,
                "f1": final_info.get("f1"),
                "pipeline": final_info.get("pipeline"),
                "invalid_count": env.invalid_count,
                "pipeline_length": len(env.pipeline_actions),
            }
        )

    return results
