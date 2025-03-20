"""
pip install pyerualjetwork
pip install gym
pip install gym[mujoco] mujoco

"""
import gym
from pyerualjetwork import ene, model_operations, data_operations
import numpy as np

# Define Genomes
input_shape = 17
output_shape = 6
population_size = 50

genome_weights, genome_activations = ene.define_genomes(input_shape, output_shape, population_size)

rewards = [0] * population_size
reward_sum = 0

generation = 0
max_generation = 50

# Training Loop Starts
for generation in range(max_generation):
    for individual in range(population_size):

        if individual == 0:
            env = gym.make('HalfCheetah-v4', render_mode='human')
            state = env.reset()
            env.model.opt.timestep = 0.003
            state = np.array(state[0])

        else:
            env.close()
            env = gym.make('HalfCheetah-v4')
            state = env.reset()
            env.model.opt.timestep = 0.003
            state = np.array(state[0])

        while True:
            # Action Calculation
            action = ene.evaluate(Input=state, weights=genome_weights[individual], activations=genome_activations[individual])
            
            action = data_operations.normalization(action, dtype=action.dtype)
            state, reward, done, truncated, _ = env.step(action)

            reward_sum += reward

            if done or truncated:
                state = env.reset()
                state = np.array(state[0])
                rewards[individual] = reward_sum
                reward_sum = 0

                break

    genome_weights, genome_activations = ene.evolver(genome_weights, genome_activations, generation, np.array(rewards, dtype=np.float32), show_info=True, activation_mutate_add_prob=0, activation_selection_add_prob=0)

model_operations.save_model(model_name='HalfCheetah_v4', model_path='HalfCheetah_v4/', W=genome_weights[0], activation_potentiation=genome_activations[0], show_architecture=True)
