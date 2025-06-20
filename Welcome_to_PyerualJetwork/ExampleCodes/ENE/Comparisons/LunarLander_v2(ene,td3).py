import gym
import numpy as np
import matplotlib.pyplot as plt
from pyerualjetwork import ene, model_ops
from pyerualjetwork.cpu import data_ops
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from multiprocessing import Pool, cpu_count


# --- ENE: ---

def evaluate_individual(args):
    individual_idx, W, A, seeds = args
    env = gym.make('LunarLander-v2')
    total_reward = 0
    model = model_ops.build_model(W=W, activations=A, model_type='MLP')

    for seed in seeds:
        state = env.reset(seed=seed)[0]
        done = truncated = False
        reward_sum = 0

        while not done and not truncated:
            action = model_ops.predict_from_memory(state, model)
            action = np.argmax(action)
            state, reward, done, truncated, _ = env.step(action)
            reward_sum += reward
        total_reward += reward_sum

    env.close()
    return total_reward / len(seeds)

env = gym.make('LunarLander-v2')
state = env.reset()[0]

input_shape = env.observation_space.shape[0]
output_shape = env.action_space.shape[0]
population_size = 300

genome_weights, genome_activations = ene.define_genomes(input_shape, output_shape, population_size, neurons=[256, 256], activation_functions=['tanh', 'tanh'])

rewards = [0] * population_size
max_rewards = []
reward_sum = 0

generation = 0
max_generation = 10

time_stamps_1 = []
max_rewards_1 = []

seeds = [random.randint(0, 2**32 - 1) for _ in range(25)]

start_time_1 = time.time()

# Training Loop Starts
for generation in range(max_generation):
    args_list = [
        (i, genome_weights[i], genome_activations[i], seeds)
        for i in range(population_size)
    ]

    with Pool(processes=cpu_count()) as pool:
        rewards = pool.map(evaluate_individual, args_list)
        
    # Evrim mekanizmasını çalıştır
    genome_weights, genome_activations = ene.evolver(
        genome_weights, genome_activations, generation, 
        np.array(rewards, dtype=np.float32), show_info=True, is_mlp=True, activation_mutate_prob=0
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
episodes = 100

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
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

env = gym.make('LunarLander-v2', render_mode='human')
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
    state = env.reset()[0]
    total_reward = 0
    step = 0

    while True:
        action = select_action(state, actor, noise=0.1)
        action = np.argmax(action)
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




plt.figure(figsize=(10, 5))
plt.plot(time_stamps_1, max_rewards_1, marker='o', linestyle='-', color='g', label="ENE")
plt.plot(time_stamps_2, max_rewards_2, marker='o', linestyle='-', color='r', label="TD3")
plt.xlabel("Time(seconds) [Lower Better]")
plt.ylabel("Max Reward [Higher Better]")
plt.title("LunarLander-v2")
plt.legend()
plt.grid()
plt.show()
