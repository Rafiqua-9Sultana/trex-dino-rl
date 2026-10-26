
from dino_env import DinoEnv


env = DinoEnv(render_mode="human")
observation, info = env.reset()

print("Dino game started!")
print("Initial observation:", observation)

try:
    for step in range(600):
        # For now, use random actions to test the game
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        if terminated or truncated:
            print("Game ended!")
            print("Final score:", info["score"])
            break
finally:
    env.close()
    print("Game closed.")
