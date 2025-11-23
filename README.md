Rocket_AI is a reinforcement learning algorithm that tries to stabilize the position of a rocket in a simplified space environment.

The concept is simple, a rocket is situated in the middle of nowhere and has only one goal, stabilize its position with the help of its three engines.

To make this model, we use Reinforcement Learning and the Q-Learning algorithm. Because we want the most optimized model, we will do Hyperparameter Optimization.

**Core Simulation**

rocket.py: The physics engine. It defines the Rocket class, calculates forces (gravity, thrust, drag), torques, and updates the physics state (position, velocity, rotation).

environment.py: The simulation world. It wraps the Rocket, manages the boundaries (screen size), and handles coordinate conversions (meters to pixels).

visualization.py: The graphics handler. It uses Pygame to draw the rocket, engine flames, and the background.
Reinforcement Learning (New)

rl_environment.py: The bridge between the physics and the AI. It simplifies the complex Environment into a standard format for the AI:
State: Angle and rotation speed.
Action: Which engines to turn on.
Reward: Points for staying stable.

agent.py: The AI brain. It contains the Deep Q-Network (DQN) neural network and the logic for the agent to learn from its mistakes and choose actions.

trainer.py: The coach. It runs multiple copies of the environment in parallel to train the agent faster and saves the best models.

**Application**
main.py: The entry point. It ties everything together. It handles the game loop and lets you switch between Manual Mode, Training Mode, and Test Mode via command line arguments.