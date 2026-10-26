
from stable_baselines3 import PPO
from dino_env import DinoEnv


def main():
    print("Loading trained Dino AI...")

    env = DinoEnv(render_mode="human")
    model = PPO.load("dino_ppo_model", env=env)

    print("AI loaded successfully!")
    print("Watch the dinosaur avoid obstacles.")

    observation, info = env.reset()
    episode_reward = 0.0

    try:
        while True:
            # Let the trained AI choose an action
            action, _states = model.predict(
                observation, deterministic=True
            )

            observation, reward, terminated, truncated, info = (
                env.step(int(action))
            )

            episode_reward += reward

            if terminated or truncated:
                print(
                    f"Episode ended | "
                    f"Score: {info['score']} | "
                    f"Reward: {episode_reward:.2f}"
                )

                observation, info = env.reset()
                episode_reward = 0.0

    except KeyboardInterrupt:
        print("\nAI game stopped by user.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
