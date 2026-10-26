
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from dino_env import DinoEnv


def main():
    print("Checking the Dino environment...")

    env = DinoEnv()
    check_env(env, warn=True)
    env.close()

    print("Environment check completed!")
    print("Starting AI training...")

    # Create the training environment
    env = DinoEnv()

    # PPO learns a policy from observations and rewards
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        verbose=1,
        seed=42,
    )

    # Train for an initial 50,000 environment steps
    model.learn(total_timesteps=50000)

    # Save the trained model
    model.save("dino_ppo_model")

    env.close()

    print("\nTraining finished!")
    print("Model saved as dino_ppo_model.zip")


if __name__ == "__main__":
    main()
