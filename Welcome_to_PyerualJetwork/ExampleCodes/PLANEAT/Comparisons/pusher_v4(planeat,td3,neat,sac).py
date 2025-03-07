import gym
import numpy as np
import matplotlib.pyplot as plt
from pyerualjetwork import planeat, data_operations
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import neat


# --- PLANEAT: ---

env = gym.make('Pusher-v4')

input_shape = env.observation_space.shape[0]
output_shape = env.action_space.shape[0]

env.close()
population_size = 50

genome_weights, genome_activations = planeat.define_genomes(input_shape, output_shape, population_size, hidden=2, neurons=[256, 256], activation_functions=['tanh', 'relu'])


rewards = [0] * population_size
max_rewards = []
reward_sum = 0

generation = 0
max_generation = 200

time_stamps_1 = []
max_rewards_1 = []


start_time_1 = time.time()

# Training Loop Starts
for generation in range(max_generation):
    for individual in range(population_size):


        if individual == 0:
            env = gym.make('Pusher-v4') # add render_mode='human' for see best individual per generation.
            state = env.reset(seed=10)

            state = np.array(state[0])

        else:
            env.close()
            env = gym.make('Pusher-v4')
            state = env.reset(seed=10)

            state = np.array(state[0])


        while True:
            # Action Calculation
            action = planeat.evaluate(Input=state, weights=genome_weights[individual], is_mlp=True, activation_potentiations=genome_activations[individual])
            
            action = data_operations.normalization(action, dtype=action.dtype)
            state, reward, done, truncated, _ = env.step(action)

            reward_sum += reward

            if done or truncated:
                state = env.reset(seed=10)
                state = np.array(state[0])
                rewards[individual] = reward_sum
                reward_sum = 0
                break

    # Evrim mekanizmasını çalıştır

    genome_weights, genome_activations = planeat.evolver(
        genome_weights, genome_activations, generation, 
        np.array(rewards, dtype=np.float32), show_info=True, is_mlp=True
    )

    max_rewards_1.append(max(rewards))
    
    # Geçen süreyi kaydet (saniye cinsinden)
    elapsed_time = time.time() - start_time_1
    time_stamps_1.append(elapsed_time)

env.close()



# --- TD3: ---



gamma = 0.99
learning_rate = 1e-3
buffer_size = 100000
batch_size = 64
tau = 0.005  
policy_noise = 0.2
noise_clip = 0.5
policy_delay = 2  
episodes = 400

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return torch.tanh(self.fc3(x))

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

        q1 = torch.tanh(self.fc1(sa))
        q1 = torch.tanh(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = torch.tanh(self.fc4(sa))
        q2 = torch.tanh(self.fc5(q2))
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
    return action

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

env = gym.make('Pusher-v4')

input_dim = env.observation_space.shape[0]
output_dim = env.action_space.shape[0]

actor = Actor(input_dim, output_dim)
critic = Critic(input_dim, output_dim)
actor_target = Actor(input_dim, output_dim)
critic_target = Critic(input_dim, output_dim)

actor_target.load_state_dict(actor.state_dict())
critic_target.load_state_dict(critic.state_dict())

actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)
replay_buffer = ReplayBuffer(buffer_size)

time_stamps_2 = []
max_rewards_2 = []

start_time_2 = time.time()

for episode in range(episodes):
    state = env.reset(seed=10)[0]
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
            next_actions = next_actions + noise

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

    elapsed_time = time.time() - start_time_2
    time_stamps_2.append(elapsed_time)
    max_rewards_2.append(total_reward)

    print(f"Episode {episode + 1}, Max Reward: {total_reward:.2f}")

env.close()


# --- SAC: ---



gamma = 0.99
learning_rate = 1e-3
buffer_size = 100000
batch_size = 64
tau = 0.005
alpha = 0.2
episodes = 400

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.log_std_layer = nn.Linear(256, output_dim)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mean = self.fc3(x)
        log_std = torch.clamp(self.log_std_layer(x), -20, 2)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        epsilon = 1e-6
        log_prob -= torch.sum(torch.log(1 - action.pow(2) + epsilon), dim=-1, keepdim=True)
        return action, log_prob

    def get_action(self, state, deterministic=True):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.forward(state)
                action = torch.tanh(mean)
            else:
                action, _ = self.sample(state)
        return action.cpu().numpy().flatten()

class Critic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Critic, self).__init__()
        self.q1 = self._build_network(input_dim, action_dim)
        self.q2 = self._build_network(input_dim, action_dim)
    
    def _build_network(self, input_dim, action_dim):
        return nn.Sequential(
            nn.Linear(input_dim + action_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

env = gym.make('Pusher-v4')

input_dim = env.observation_space.shape[0]
output_dim = env.action_space.shape[0]

actor = Actor(input_dim, output_dim)
critic = Critic(input_dim, output_dim)
critic_target = Critic(input_dim, output_dim)
critic_target.load_state_dict(critic.state_dict())

actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)
replay_buffer = ReplayBuffer(buffer_size)

time_stamps_4 = []
max_rewards_4 = []
start_time = time.time()

for episode in range(episodes):
    state = env.reset(seed=10)[0]
    total_reward = 0
    step = 0
    while True:
        action = actor.get_action(state, deterministic=False)
        action = data_operations.normalization(action, dtype=action.dtype)
        next_state, reward, done, truncated, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done or truncated:
            break

        if len(replay_buffer) > batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            
            states = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones).unsqueeze(1)

            with torch.no_grad():
                next_actions, next_log_probs = actor.sample(next_states)
                q1_target, q2_target = critic_target(next_states, next_actions)
                q_target_min = torch.min(q1_target, q2_target)
                q_target = rewards + gamma * (1 - dones) * (q_target_min - alpha * next_log_probs)
            
            q1, q2 = critic(states, actions)
            critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
            
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            actions_new, log_probs = actor.sample(states)
            q1_new, q2_new = critic(states, actions_new)
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (alpha * log_probs - q_new).mean()
            
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            
            soft_update(critic_target, critic, tau)
        step += 1
    
    elapsed_time = time.time() - start_time
    time_stamps_4.append(elapsed_time)
    max_rewards_4.append(total_reward)
    
    print(f"Episode {episode + 1}, Max Reward: {total_reward:.2f}")


plt.figure(figsize=(10, 5))
plt.plot(time_stamps_1, max_rewards_1, marker='o', linestyle='-', color='g', label="PLANEAT (3 Hyperparameters Tuned)")
plt.plot(time_stamps_2, max_rewards_2, marker='o', linestyle='-', color='r', label="TD3 (13 Hyperparameters Tuned)")
plt.plot(time_stamps_4, max_rewards_4, marker='o', linestyle='-', color='y', label="SAC (11 Hyperparameters Tuned)")
plt.xlabel("Time(seconds) [Lower Better]")
plt.ylabel("Max Reward [Higher Better]")
plt.title("Pusher-v4")
plt.legend()
plt.grid()
plt.show()
