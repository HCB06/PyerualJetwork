import gym
import numpy as np
from pyerualjetwork.cpu import data_ops
from pyerualjetwork import ene, model_ops
import numpy as np
import random


# --- ENE(Eugenic Neuroevolution): ---


env = gym.make('LunarLanderContinuous-v2')
state = env.reset()[0]

input_shape = env.observation_space.shape[0]
output_shape = env.action_space.shape[0]
population_size = 300

genome_weights, genome_activations = ene.define_genomes(input_shape, output_shape, population_size, neurons=[256, 256], activation_functions=['tanh', 'tanh'])


rewards = [0] * population_size
max_rewards = []
reward_sum = 0

generation = 0
max_generation = 30

time_stamps_1 = []
max_rewards_1 = []
seeds = []

for _ in range(10):
    seeds.append(random.randint(0, 2**32 - 1))

""" TRAINING: """

# Training Loop Starts
for generation in range(max_generation):

    for _ in range(10):
        current_seed = seeds[_]
        for individual in range(population_size):

            while True:
                # Action Calculation
                model = model_ops.build_model(W=genome_weights[individual], activations=genome_activations[individual], model_type='MLP')
                action = model_ops.predict_from_memory(state, model)
                
                action = data_ops.normalization(action, dtype=action.dtype)
                state, reward, done, truncated, _ = env.step(action)
                reward_sum += reward

                if done or truncated:
                    state = env.reset(seed=current_seed)
                    state = np.array(state[0])
                    rewards[individual] += reward_sum
                    reward_sum = 0
                    break
    
    seeds = []
    for _ in range(10):
        seeds.append(random.randint(0, 2**32 - 1))

    model = model_ops.build_model(W=genome_weights[np.argmax(rewards)], activations=genome_activations[np.argmax(rewards)], model_type='MLP')
    model_ops.save_model(model, model_name=f'lunar_expert_gen{generation}')

    # Evrim mekanizmasını çalıştır
    genome_weights, genome_activations = ene.evolver(
        genome_weights, genome_activations, generation, 
        np.array(rewards.copy(), dtype=np.float32), show_info=True, is_mlp=True
    )
    max_rewards_1.append(max(rewards))
    
    rewards = [0] * population_size

env.close()

model_ops.save_model(model_name='lunar_expert_last', model_type='MLP', W=genome_weights[np.argmax(rewards)], activation_potentiation=genome_activations[np.argmax(rewards)])


""" TESTING IN 100 EPISODE: """

"""
#NOTE: If you want test another model use this:

model = model_ops.load_model(model_name='lunar_expert_gen13', model_path='')

genome_activations = model.activations
genome_weights = model.weights

"""

env = gym.make('LunarLanderContinuous-v2', render_mode='human')
state = env.reset()
state = np.array(state[0])
reward_sum = 0
rewards = [0] * 100

for i in range(100):

    while True:

        # Action Calculation
        model = model_ops.build_model(W=genome_weights[np.argmax(rewards)], activations=genome_activations[np.argmax(rewards)], model_type='MLP')
        action = model_ops.predict_from_memory(state, model)
        
        
        action = data_ops.normalization(action, dtype=action.dtype)
        state, reward, done, truncated, _ = env.step(action)

        reward_sum += reward

        if done or truncated:
            state = env.reset()
            state = np.array(state[0])
            rewards[i] = reward_sum
            reward_sum = 0
            print(f'return: {rewards[i]}')
            break

print(f'average return: {np.mean(np.array(rewards))}')
