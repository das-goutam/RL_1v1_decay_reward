# Defender-Attacker Differential Game

A reinforcement learning implementation of a two-player zero-sum differential game where defender and attacker agents learn optimal strategies simultaneously. This repository contains code for training and evaluating agents in a pursuit-evasion scenario with visibility-based mechanics.

## Overview

In this differential game:
- **Attacker**: Tries to reach a target while avoiding capture
- **Defender**: Tries to either capture the attacker or prevent it from reaching the target
- **Game Mechanics**: Include line-of-sight blocking, Cartesian ovals for reachable sets, and time-to-target optimality conditions

Both agents use Soft Actor-Critic (SAC) algorithm with specialized reward shaping to learn optimal strategies.

## Features

- **Multi-agent Reinforcement Learning**: Both defender and attacker learn simultaneously
- **High-performance Implementation**: Optimized with parallel environments, vectorized operations, and JIT compilation
- **Visualization Tools**: Comprehensive visualization of agent strategies and game dynamics
- **Hot-start Training**: Continue training from previously trained models
- **Analytics**: Detailed performance metrics and trajectory analysis

## Requirements

- Python 3.8+
- PyTorch 1.8+
- NumPy
- Matplotlib
- Gym
- SciPy
- tqdm
- Numba

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/defender-attacker-game.git
cd defender-attacker-game

# Create a virtual environment (optional)
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
# Train from scratch for 5000 episodes
python train_1v1.py --episodes 5000
```

### Hot-start Training

Continue training from the latest saved models:

```bash
# Continue training for a total of 5000 episodes (if previously trained for 2000, will train for 3000 more)
python train_hotstart.py --episodes 5000 --hotstart
```

### Testing and Visualization

Run test episodes with trained models:

```bash
# Test agents with the latest trained models
python test_1v1.py --episodes 10

# Test with specific model files
python test_1v1.py --attacker_model models/attacker_agent_20240304_episode_2000.pt --defender_model models/defender_agent_20240304_episode_2000.pt
```

## Command Line Options

### Training (train_1v1.py / train_hotstart.py)

| Option | Description | Default |
|--------|-------------|---------|
| `--episodes` | Number of episodes to train | 5000 |
| `--batch_size` | Batch size for training | 256 |
| `--buffer_size` | Replay buffer size | 1000000 |
| `--eval_interval` | Evaluation interval | 100 |
| `--save_interval` | Model save interval | 100 |
| `--vis_interval` | Visualization interval | 500 |
| `--model_dir` | Directory for models | models |
| `--log_dir` | Directory for logs | logs |
| `--num_envs` | Number of parallel environments | 4 |
| `--seed` | Random seed | 50 |
| `--gpu` | GPU ID to use | 0 |

### Hot-start specific options

| Option | Description | Default |
|--------|-------------|---------|
| `--hotstart` | Enable hot-start from latest models | True |
| `--no-hotstart` | Disable hot-start | - |

### Testing (test_1v1.py)

| Option | Description | Default |
|--------|-------------|---------|
| `--model_dir` | Directory containing model files | models |
| `--episodes` | Number of test episodes to run | 5 |
| `--render` | Enable visualization | True |
| `--no-render` | Disable visualization | - |
| `--gamma` | Defender speed relative to attacker | 0.5 |
| `--attacker_model` | Specific attacker model to load | (latest) |
| `--defender_model` | Specific defender model to load | (latest) |

## Environment Parameters

- `gamma`: Defender speed relative to attacker (0 < gamma < 1)
- `dt`: Time step for simulation
- `r_capture`: Capture radius for the defender

## Terminal Conditions

The game can end in several ways:
- `attacker_reached_target`: Attacker successfully reaches the target
- `defender_reached_target`: Defender reaches the target before the attacker
- `attacker_captured`: Defender captures the attacker
- `defender_time_advantage`: Defender has guaranteed interception capability
- `unblocked_path_advantage`: Attacker has unblocked path to target with time advantage
- `target_inside_oval`: Target is inside the Cartesian oval (defender advantage)
- `timeout`: Episode exceeds maximum steps

## Project Structure

```
defender-attacker-game/
├── models/                   # Saved model files
├── logs/                     # Training logs
├── plots/                    # Visualization outputs
├── test_results/             # Test results and analytics
├── train_1v1.py              # Main training script
├── train_hotstart.py         # Training with hot-start capability
├── test_1v1.py               # Testing and visualization script
├── lookup_table_value.mat    # Lookup table for analytical solutions
└── requirements.txt          # Project dependencies
```

## How Training Works

1. Both agents use SAC with shaped rewards
2. Replay buffers store experiences for both agents
3. Training uses parallel environments for faster data collection
4. Models are periodically saved and evaluated
5. Reward decay is applied as training progresses to stabilize learning
6. Visualizations are created to monitor learning progress

## Acknowledgements

This work is based on research in differential games and pursuit-evasion scenarios. Special thanks to contributors and researchers in the field of game theory and reinforcement learning.

## License

[MIT License](LICENSE)
