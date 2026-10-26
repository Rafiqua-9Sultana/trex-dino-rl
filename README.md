# T-Rex Dino Bot Using Reinforcement Learning

## Project Overview
This project implements an AI-powered version of the Chrome Dino game using Reinforcement Learning. The agent learns to avoid obstacles by interacting with a custom game environment and receiving rewards.

## Objectives
- Develop a custom Dino game environment using Python.
- Train an AI agent using Proximal Policy Optimization (PPO).
- Enable the agent to play the game automatically.
- Evaluate the trained model across multiple episodes.

## Technologies Used
- Python
- Gymnasium
- Stable-Baselines3
- PPO (Proximal Policy Optimization)
- Pygame
- NumPy

## How It Works
1. The game environment provides observations about the dinosaur and approaching obstacles.
2. The PPO agent selects an action, such as running or jumping.
3. The environment returns a reward based on the agent's actions and game progress.
4. The agent learns through repeated interactions.
5. The trained model is used to play the game automatically.

## Project Files
- `dino_env.py` — Custom game environment.
- `train.py` — Trains the reinforcement learning agent.
- `play.py` — Runs the trained AI.
- `evaluate.py` — Evaluates the agent's performance.
- `requirements.txt` — Lists the required Python packages.
- `dino_ppo_model.zip` — Saved trained model.

## Installation and Setup

### 1. Clone the repository
```bash
git clone https://github.com/Rafiqua-9Sultana/trex-dino-rl.git
cd trex-dino-rl
```

### 2. Create a virtual environment
```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 5. Train the AI
```bash
python train.py
```

### 6. Play using the trained AI
```bash
python play.py
```

### 7. Evaluate the AI
```bash
python evaluate.py
```

## Current Evaluation Results
The trained model was tested across 10 episodes.

- Average score: 1.4
- Best score: 3
- Average reward: 41.44
- Average episode length: 375.4 steps

These are the results from the current evaluation run. The agent successfully runs, but its obstacle-avoidance performance still needs improvement.

## Future Improvements
- Improve obstacle-avoidance accuracy.
- Experiment with longer training runs and PPO hyperparameters.
- Add a visual score display and improved game graphics.
- Compare different reinforcement learning settings.

## Author
Rafiqua Sultana
