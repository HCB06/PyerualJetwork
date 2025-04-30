import gym
import numpy as np
from pyerualjetwork.cpu import data_ops
from pyerualjetwork import ene, model_ops
import time

# --- ENE WITH MLP: ---


input_shape = 17
output_shape = 6
population_size = 50

genome_weights, genome_activations = ene.define_genomes(input_shape, output_shape, population_size, neurons=[256, 128], activation_functions=['tanh', 'tanh'])

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
            model = model_ops.build_model(W=genome_weights[individual], activations=genome_activations[individual], model_type='MLP')
            action = model_ops.predict_from_memory(state, model)

            action = data_ops.normalization(action, dtype=action.dtype)
            state, reward, done, truncated, _ = env.step(action)

            reward_sum += reward

            if done or truncated:
                state = env.reset()
                state = np.array(state[0])
                rewards[individual] = reward_sum
                reward_sum = 0
                break

    max_rewards_1.append(max(rewards))
    genome_weights, genome_activations = ene.evolver(
    genome_weights, genome_activations, generation, 
    np.array(rewards, dtype=np.float32), show_info=True, is_mlp=True
    )
    
model = model_ops.build_model(W=genome_weights[np.argmax(rewards)], activations=genome_activations[np.argmax(rewards)], model_type='MLP')
model_ops.save_model(model, model_name='cita')

env.close()
env = gym.make('HalfCheetah-v4', render_mode='human')
env.model.opt.timestep = 0.003
state = env.reset()[0]

while True:
    # Action Calculation
    action = model_ops.predict_from_storage(Input=state, model_path='', model_name='cita')

    action = data_ops.normalization(action, dtype=action.dtype)
    state, reward, done, truncated, _ = env.step(action)

    reward_sum += reward

    if done or truncated:
        state = env.reset()
        state = np.array(state[0])
        rewards[individual] = reward_sum
        reward_sum = 0