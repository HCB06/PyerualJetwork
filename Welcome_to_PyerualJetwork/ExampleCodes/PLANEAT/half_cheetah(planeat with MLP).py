import gym
import numpy as np
from pyerualjetwork import planeat, data_operations, model_operations
import time


# --- PLANEAT WITH MLP: ---


input_shape = 17
output_shape = 6
population_size = 50

genome_weights, genome_activations = planeat.define_genomes(input_shape, output_shape, population_size, hidden=2, neurons=[256, 128], activation_functions=['tanh', 'tanh'])

rewards = [0] * population_size
max_rewards = []
reward_sum = 0

generation = 0
max_generation = 200

time_stamps_1 = []
max_rewards_1 = []

env = gym.make('HalfCheetah-v4')
env.model.opt.timestep = 0.003
state = env.reset()[0]

start_time_1 = time.time()

# Training Loop Starts
for generation in range(max_generation):
    for individual in range(population_size):
        while True:
            # Action Calculation
            action = planeat.evaluate(Input=state, weights=genome_weights[individual], activation_potentiations=genome_activations[individual], is_mlp=True)

            action = data_operations.normalization(action, dtype=action.dtype)
            state, reward, done, truncated, _ = env.step(action)

            reward_sum += reward

            if done or truncated:
                state = env.reset()
                state = np.array(state[0])
                rewards[individual] = reward_sum
                reward_sum = 0
                break

    max_rewards_1.append(max(rewards))

model_operations.save_model(model_name='cita', model_type='MLP', W=genome_weights[0], activation_potentiation=genome_activations[0])

env.close()
env = gym.make('HalfCheetah-v4', render_mode='human')
env.model.opt.timestep = 0.003
state = env.reset()[0]

while True:
    # Action Calculation
    action = model_operations.predict_model_ssd(Input=state, model_path='', model_name='cita')

    action = data_operations.normalization(action, dtype=action.dtype)
    state, reward, done, truncated, _ = env.step(action)

    reward_sum += reward

    if done or truncated:
        state = env.reset()
        state = np.array(state[0])
        rewards[individual] = reward_sum
        reward_sum = 0