print("Starting DDPG training...")  # Add this at the beginning

import os
print(f"Script started with PID: {os.getpid()}")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from arm261_env import Arm261Env

print("Libraries imported successfully!")  # Debug print


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256) #First hidden layer
        self.layer2 = nn.Linear(256, 256) #Second hidden layer
        self.layer3 = nn.Linear(256, action_dim) # Output layer
        self.max_action = max_action #scaling factor for action
    
    def forward(self, state):
        x = torch.relu(self.layer1(state)) #Apply ReLu activation
        x = torch.relu(self.layer2(x)) #Apply Relu activation
        x = torch.tanh(self.layer3(x)) * self.max_action #Apply Tanh and scale output
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 256) # First hidden layer
        self.layer2 = nn.Linear(256, 256) # Second hidden layer
        self.layer3 = nn.Linear(256, 1) # Output layer
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1) # Concatenate state and action
        x = torch.relu(self.layer1(x)) # First hidden layer with ReLU
        x = torch.relu(self.layer2(x)) # Second hidden layer with ReLU
        return self.layer3(x) # Output Q-value

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005, lr=1e-3):
        '''Gamma (γ): Discount factor for future rewards.
            Tau (τ): Soft update factor for target networks.
            Learning Rate (lr): Controls step size for optimization.'''
        
        self.actor = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim, max_action)
        self.target_critic = Critic(state_dim, action_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.gamma = gamma
        self.tau = tau
    #takes state as input, predicts action, and converts it to NumPy array.
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self, replay_buffer, batch_size=64):
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        next_state = torch.FloatTensor(next_state)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        done = torch.FloatTensor(done).unsqueeze(1)
        
        # Compute target Q value
        next_action = self.target_actor(next_state)
        target_Q = self.target_critic(next_state, next_action)
        target_Q = reward + (1 - done) * self.gamma * target_Q.detach()
        
        # Update critic
        current_Q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# Setup environment
env = Arm261Env()

print("Environment initialized!")  # Debug print
#Loads the custom MuJoCo environment 
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]
'''Extracts: State dimension (number of inputs).
            Action dimension (number of outputs).
            Max action value (scaling factor).'''

#Creates the DDPG agent.
ddpg_agent = DDPGAgent(state_dim, action_dim, max_action)
print("DDPG Agent initialized!")  # Debug print



import mujoco.viewer  # Import the MuJoCo viewer
viewer = mujoco.viewer.launch_passive(env.model, env.data)
import sys
sys.path.append("D:/park2")  # Ensure Python finds replay_buffer.py
from replay_buffer import ReplayBuffer  # Import the Replay Buffer
# Initialize Replay Buffer
replay_buffer = ReplayBuffer(max_size=100000, state_dim=state_dim, action_dim=action_dim)




#Training loop
mse_history = []  # Store MSE values per episode

#Resets the environment at the start of each episode.
#Tracks total reward and total jerk (movement smoothness).
for episode in range(10):
    state = env.reset()
    episode_reward = 0
    total_jerk = 0
    done = False

    # Store data for analysis
    actions_taken = []  
    positions = []  
    episode_mse = []

    while not done:
        action = ddpg_agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward

        jerk = env.compute_jerk()  # Compute jerk
        total_jerk += jerk  # Track total jerk

        # Store action taken
        actions_taken.append(action)

        # Get & store end-effector position
        end_effector_pos = env.get_end_effector_position()  
        positions.append(end_effector_pos)

        # Compute & store MSE
        target_pos = env.get_target_position()  
        mse = np.mean((end_effector_pos - target_pos) ** 2) #using mean 
        episode_mse.append(mse)

        # Store experience in replay buffer
        replay_buffer.add(state, action, next_state, reward, done)

        # Train the agent if buffer has enough samples
        if replay_buffer.size > 64:  
            ddpg_agent.train(replay_buffer, batch_size=64)

        state = next_state
        viewer.sync()

    # Store final MSE for plotting
    mse_history.append(np.mean(episode_mse))  

    # Print episode results
    print(f"Episode {episode + 1}: Reward: {episode_reward:.2f}, Avg Jerk: {total_jerk / max(1, len(actions_taken)):.4f}")
    #print("Actions taken:", actions_taken)
    #print("End-effector positions:", positions)
    #print("MSE values:", episode_mse)

print("Training complete. Press Ctrl+C to exit viewer.")

# Keep viewer running
while viewer.is_running():
    viewer.sync()

# Plot MSE vs Episodes
import matplotlib.pyplot as plt

plt.plot(range(1, len(mse_history) + 1), mse_history, marker='o', linestyle='-', color='b', label='MSE')
plt.xlabel("Episode")
plt.ylabel("MSE (End-Effector vs Target)")
plt.title("MSE of End-Effector Position Over Training")
plt.legend()
plt.grid(True)
plt.show()

# Close viewer **after training is done**
env.plot_jerk()
env.plot_mjt_comparison()  # Plot final MJT analysis for all episodes



