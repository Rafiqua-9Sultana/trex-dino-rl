
from stable_baselines3 import PPO
from dino_env import DinoEnv


def main():
    print("Loading trained Dino AI for evaluation...")

    # Run without opening a game window
    env = DinoEnv(render_mode=None)
    model = PPO.load("dino_ppo_model")

    scores = []
    rewards = []
    episode_lengths = []

    number_of_episodes = 10

    for episode in range(1, number_of_episodes + 1):
        observation, info = env.reset()

        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(
                observation,
                deterministic=True
            )

            observation, reward, terminated, truncated, info = (
                env.step(int(action))
            )

            total_reward += float(reward)
            steps += 1

        score = info.get("score", 0)

        scores.append(score)
        rewards.append(total_reward)
        episode_lengths.append(steps)

        print(
            f"Episode {episode:02d} | "
            f"Score: {score} | "
            f"Reward: {total_reward:.2f} | "
            f"Steps: {steps}"
        )

    print("\n========== EVALUATION SUMMARY ==========")
    print(f"Episodes tested: {number_of_episodes}")
    print(f"Average score: {sum(scores) / len(scores):.2f}")
    print(f"Best score: {max(scores)}")
    print(f"Average reward: {sum(rewards) / len(rewards):.2f}")
    print(
        f"Average episode length: "
        f"{sum(episode_lengths) / len(episode_lengths):.2f} steps"
    )

    env.close()


if __name__ == "__main__":
    main()
