# 🚀 Rocket AI

**Rocket AI** is a reinforcement learning project designed to teach an AI agent how to stabilize a rocket in a physics-based environment. Using **Deep Q-Learning (DQN)**, the agent learns to control three thrusters to maintain stability against gravity and external forces.

The project is optimized for performance, featuring **parallel environment simulation** and hardware acceleration on **Apple Silicon (MPS)**.

## 📋 Table of Contents
- [About](#about)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies](#technologies)

## 📖 About
The concept is simple: a rocket is situated in a 2D space and has one goal—stabilize its orientation and position using its three engines. 

Instead of hard-coding control logic, we use **Reinforcement Learning**. The agent observes its state (angle, angular velocity) and learns the optimal firing sequence for its engines through trial and error, maximizing a reward function based on stability.

## ✨ Features
- **Physics Engine**: Custom 2D physics simulation including gravity, drag, torque, and moment of inertia.
- **Deep Q-Network (DQN)**: Implements Experience Replay and Target Networks for stable learning.
- **Parallel Training**: Runs multiple simulation environments simultaneously to accelerate data collection.
- **Hardware Acceleration**: Automatically utilizes Apple Silicon (MPS) or CUDA if available.
- **Real-time Visualization**: Built with Pygame to visualize the rocket's behavior and the agent's performance.

## 🛠 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Rocket_AI.git
   cd Rocket_AI
   ```

2. **Install Dependencies**
   Ensure you have Python 3.11+ installed.
   ```bash
   pip install numpy pygame torch
   ```

## 🎮 Usage

The application `src/main.py` is the entry point for all modes.

### 1. Manual Control
Try to control the rocket yourself!
```bash
python src/main.py
```
- **Controls**:
    - `I`: Left Engine
    - `O`: Center Engine
    - `P`: Right Engine
    - `Q`: Quit
    - `V`: Toggle Visualization

### 2. Train the AI
Train the DQN agent. This will run multiple environments in parallel and save checkpoints.
```bash
python src/main.py --mode train
```

### 3. Test the AI
Watch the trained agent control the rocket.
```bash
python src/main.py --mode test --model final_model.pth
```

## 📂 Project Structure

### Core Simulation
- **`src/rocket.py`**: The physics engine. Defines the `Rocket` class, calculating forces (gravity, thrust, drag), torques, and updating physics state.
- **`src/environment.py`**: The simulation world. Wraps the rocket, manages boundaries, and handles coordinate conversions.
- **`src/visualization.py`**: Graphics handler using Pygame to render the rocket and environment.

### Reinforcement Learning
- **`src/rl_environment.py`**: RL Wrapper. Converts the raw physics environment into a standard RL format (State, Action, Reward).
    - **State**: `[cos(θ), sin(θ), angular_velocity]`
    - **Action**: 8 discrete engine combinations.
- **`src/agent.py`**: The AI Brain. Contains the Deep Q-Network (DQN) architecture and learning logic.
- **`src/trainer.py`**: The Coach. Manages parallel training loops and synchronizes the agent.

### Application
- **`src/main.py`**: Entry point. Handles CLI arguments and switches between Manual, Train, and Test modes.

## 💻 Technologies
- **Language**: Python
- **ML Framework**: PyTorch
- **Visualization**: Pygame
- **Math/Physics**: NumPy