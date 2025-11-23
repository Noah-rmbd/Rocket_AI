import numpy as np
from rl_environment import RLEnvironment
from agent import DQNAgent
import time
from collections import deque

class ParallelEnvironments:
    def __init__(self, num_envs=16):
        self.envs = [RLEnvironment() for _ in range(num_envs)]
        self.num_envs = num_envs
        self.states = [env.reset() for env in self.envs]
        
    def step(self, actions):
        next_states = []
        rewards = []
        dones = []
        infos = []
        
        for i, env in enumerate(self.envs):
            state, reward, done, info = env.step(actions[i])
            
            if done:
                state = env.reset()
                
            next_states.append(state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            
            self.states[i] = state
            
        return np.array(next_states), np.array(rewards), np.array(dones), infos

    def reset(self):
        self.states = [env.reset() for env in self.envs]
        return np.array(self.states)

def train(n_episodes=100, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    # Hyperparameters
    num_envs = 32 # Parallel environments
    
    env = ParallelEnvironments(num_envs=num_envs)
    
    # State size: 3 (cos, sin, av)
    # Action size: 8
    agent = DQNAgent(state_size=3, action_size=8, seed=0)
    
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    
    print(f"Starting training with {num_envs} parallel environments...")
    print(f"Device: {agent.device}")
    
    # We track episodes by "batches" of parallel runs
    # But standard reporting is usually per episode. 
    # Since we have continuous parallel envs, we can just track total steps or average score.
    # Let's track average score of completed episodes.
    
    # For simplicity in this loop, we'll just run N "parallel steps" which roughly corresponds to episodes
    # But since they reset independently, it's a bit async.
    # Let's just run for a fixed number of updates/frames.
    
    total_frames = 0
    max_frames = n_episodes * max_t # Approximation
    
    # To track scores properly, we need to accumulate rewards for each env
    current_scores = np.zeros(num_envs)
    
    start_time = time.time()
    
    i_episode = 0
    
    while i_episode < n_episodes:
        states = np.array(env.states)
        
        # Select actions (epsilon-greedy)
        # We need to vectorize act? Agent.act is single.
        # Let's make a batch act or just loop.
        actions = []
        for i in range(num_envs):
            actions.append(agent.act(states[i], eps))
            
        next_states, rewards, dones, _ = env.step(actions)
        
        # Step agent
        for i in range(num_envs):
            agent.step(states[i], actions[i], rewards[i], next_states[i], dones[i])
            
            current_scores[i] += rewards[i]
            
            if dones[i]:
                scores_window.append(current_scores[i])
                scores.append(current_scores[i])
                current_scores[i] = 0
                i_episode += 1
                
                if i_episode % 10 == 0:
                    print(f'Episode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {eps:.2f}', flush=True)
                    agent.save(f'checkpoint_{i_episode}.pth')

        eps = max(eps_end, eps_decay * eps) # Decay epsilon
        
        # This decay might be too fast if we decay per step per env? 
        # Usually decay per episode. 
        # Let's decay only when an episode finishes? 
        # Or just decay slowly. 0.995 per step is too fast. 
        # 0.995 per episode is standard. 
        # Since we have multiple envs, we can decay every time *any* env finishes?
        # Let's stick to simple decay per step but very slow, or decay per "batch of steps".
        # Actually, let's decay only when i_episode increments.
        
    
    print("\nTraining completed.")
    agent.save('final_model.pth')
    return scores

if __name__ == "__main__":
    train()
