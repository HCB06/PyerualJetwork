import gym
import numpy as np
from pyerualjetwork import ene_cuda, data_operations_cuda, model_operations_cuda
import cupy as cp
import random

# --- ENE(Eugenic Neuroevolution): ---

env = gym.make('ALE/Galaxian-v5', frameskip=1)
env = gym.wrappers.AtariPreprocessing(
    env, 
    frame_skip=4,
    screen_size=84,
    grayscale_obs=True,
    grayscale_newaxis=False,
    scale_obs=True
)
env = gym.wrappers.FrameStack(env, 4)
state = env.reset()[0]
state = np.array(state).ravel().reshape(1, -1)

input_shape = state.shape[1]
output_shape = env.action_space.n

population_size = 50
genome_weights, genome_activations = ene_cuda.define_genomes(input_shape, output_shape, population_size, neurons=[256, 256], activation_functions=['tanh', 'tanh'])
rewards = [0] * population_size
max_rewards = []
reward_sum = 0
generation = 0
max_generation = 30
seed = 0

# Training Loop Starts
for generation in range(max_generation):

    for individual in range(population_size):
            i = 0
            if individual == 0:
                env = gym.make('ALE/Galaxian-v5', frameskip=1, render_mode='human')
                env = gym.wrappers.AtariPreprocessing(
                    env, 
                    frame_skip=4, 
                    screen_size=84,
                    grayscale_obs=True,
                    grayscale_newaxis=False,
                    scale_obs=True
                )
                env = gym.wrappers.FrameStack(env, 4)
                state = env.reset(seed=seed)[0]
                state = np.array(state).ravel().reshape(1, -1)
            else:
                env.close()
                env = gym.make('ALE/Galaxian-v5', frameskip=1)
                env = gym.wrappers.AtariPreprocessing(
                    env, 
                    frame_skip=4, 
                    screen_size=84,
                    grayscale_obs=True,
                    grayscale_newaxis=False,
                    scale_obs=True
                )
                env = gym.wrappers.FrameStack(env, 4)
                state = env.reset(seed=seed)[0]
                state = np.array(state).ravel().reshape(1, -1)
            
            while True:
                # Action Calculation
                action = ene_cuda.evaluate(Input=cp.array(state, dtype=cp.float32), is_mlp=True, weights=genome_weights[individual], activations=genome_activations[individual])
                
                action = data_operations_cuda.normalization(action, dtype=action.dtype)
                action = np.argmax(action)
                next_state, reward, done, truncated, _ = env.step(int(action))
                next_state = np.array(next_state).ravel().reshape(1, -1)
                reward_sum += reward
                state = next_state
                i += 1
                if done or truncated or i == 1000:
                    state = env.reset(seed=seed)[0]
                    state = np.array(state[0]).ravel().reshape(1, -1)
                    rewards[individual] = reward_sum
                    reward_sum = 0
                    break


    model_operations_cuda.save_model(model_name=f'galaxian_gen{generation}', model_type='MLP', W=genome_weights[0], activation_potentiation=genome_activations[0])

    genome_weights, genome_activations = ene_cuda.evolver(
        genome_weights, genome_activations, generation,
        np.array(rewards, dtype=np.float32), show_info=True, is_mlp=True
    )


model_operations_cuda.save_model(model_name='galaxian_last', model_type='MLP', W=genome_weights[0], activation_potentiation=genome_activations[0])
env.close()


# --- TESTING THE MODEL ---
model = model_operations_cuda.load_model(model_name='galaxian_last', model_path='')

genome_activations = model[model_operations_cuda.get_act_pot()]
genome_weights = model[model_operations_cuda.get_weights()]


env = gym.make('ALE/Galaxian-v5', frameskip=1, render_mode='human')
env = gym.wrappers.AtariPreprocessing(
    env, 
    frame_skip=4, 
    screen_size=84,
    grayscale_obs=True,
    grayscale_newaxis=False,
    scale_obs=True
)
env = gym.wrappers.FrameStack(env, 4)
state = env.reset(seed=seed)[0]
state = np.array(state).ravel().reshape(1, -1)
reward_sum = 0
rewards = [0] * 100
for i in range(100):
    while True:
        # Action Calculation
        action = ene_cuda.evaluate(Input=cp.array(state, dtype=cp.float32), is_mlp=True, weights=genome_weights, activation_potentiations=genome_activations)
        
        action = data_operations_cuda.normalization(action, dtype=action.dtype)
        action = np.argmax(action)
        next_state, reward, done, truncated, _ = env.step(int(action))
        next_state = np.array(next_state).ravel().reshape(1, -1)
        reward_sum += reward
        state = next_state

        if done or truncated:
            state = env.reset(seed=seed)[0]
            state = np.array(state).ravel().reshape(1, -1)
            rewards[i] = reward_sum
            reward_sum = 0
            print(f'return: {rewards[i]}')

            break
    
print(f'average return: {np.mean(np.array(rewards))}')
input()
