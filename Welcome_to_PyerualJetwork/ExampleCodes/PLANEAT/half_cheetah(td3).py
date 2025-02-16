import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from pyerualjetwork import data_operations

gamma = 0.99
learning_rate = 1e-3
buffer_size = 100000
batch_size = 64
tau = 0.005  
policy_noise = 0.2
noise_clip = 0.5
policy_delay = 2  
episodes = 200

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.max_action = max_action
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x)) * self.max_action

# Define the Critic Network (Twin Critics)
class Critic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        self.fc4 = nn.Linear(input_dim + action_dim, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 1)
    
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)

        q1 = torch.relu(self.fc1(sa))
        q1 = torch.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = torch.relu(self.fc4(sa))
        q2 = torch.relu(self.fc5(q2))
        q2 = self.fc6(q2)

        return q1, q2

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def select_action(state, actor, noise=0.1):
    state = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        action = actor(state).cpu().numpy().flatten()
    action += np.random.normal(0, noise, size=action.shape)
    return np.clip(action, -max_action, max_action)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

env = gym.make('HalfCheetah-v4', render_mode='human')
env.model.opt.timestep = 0.003
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]

actor = Actor(input_dim, output_dim, max_action)
critic = Critic(input_dim, output_dim)
actor_target = Actor(input_dim, output_dim, max_action)
critic_target = Critic(input_dim, output_dim)

actor_target.load_state_dict(actor.state_dict())
critic_target.load_state_dict(critic.state_dict())

actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)
replay_buffer = ReplayBuffer(buffer_size)

for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    step = 0

    while True:
        action = select_action(state, actor, noise=0.1)
        action = data_operations.normalization(action, dtype=action.dtype)
        next_state, reward, done, truncated, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done or truncated:
            break

        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones).unsqueeze(1)

            next_actions = actor_target(next_states)
            noise = torch.clamp(torch.randn_like(next_actions) * policy_noise, -noise_clip, noise_clip)
            next_actions = torch.clamp(next_actions + noise, -max_action, max_action)

            q1_target, q2_target = critic_target(next_states, next_actions)
            q_target = rewards + gamma * torch.min(q1_target, q2_target) * (1 - dones)

            q1, q2 = critic(states, actions)
            critic_loss = nn.MSELoss()(q1, q_target.detach()) + nn.MSELoss()(q2, q_target.detach())

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            if step % policy_delay == 0:
                actor_loss = -critic(states, actor(states))[0].mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                soft_update(actor_target, actor, tau)
                soft_update(critic_target, critic, tau)

        step += 1

    print(f"Episode {episode + 1}, Max Reward: {total_reward:.2f}")

env.close()

torch.save(actor.state_dict(), "td3_actor_halfcheetah.pth")
torch.save(critic.state_dict(), "td3_critic_halfcheetah.pth")