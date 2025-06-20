import gym
import numpy as np
from pyerualjetwork.cuda import data_ops
from pyerualjetwork import ene, model_ops
from multiprocessing import Pool, cpu_count


# --- ENE(Eugenic Neuroevolution): ---

def evaluate_individual(args):
    individual_idx, W, A, seeds = args
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
    state = env.reset(seed=0)[0]
    state = np.array(state[0]).ravel().reshape(1, -1)
    total_reward = 0
    model = model_ops.build_model(W=W, activations=A, model_type='MLP')
    done = truncated = False
    reward_sum = 0

    while not done and not truncated:
        action = model_ops.predict_from_memory(state, model, cuda=True)
        action = data_ops.normalization(action)
        action = np.argmax(action)
        next_state, reward, done, truncated, _ = env.step(int(action))
        next_state = np.array(next_state[0]).ravel().reshape(1, -1)
        reward_sum += reward
        state = next_state

    total_reward += reward_sum

    env.close()
    return total_reward

def main():
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
    state = env.reset(seed=0)[0]
    state = np.array(state[0]).ravel().reshape(1, -1)

    input_shape = state.shape[1]
    output_shape = env.action_space.n

    population_size = 50
    genome_weights, genome_activations = ene.define_genomes(input_shape, output_shape, population_size, neurons=[256, 256], activation_functions=['tanh', 'tanh'])
    rewards = [0] * population_size
    reward_sum = 0
    generation = 0
    max_generation = 30
    seed = 0

    # Training Loop Starts
    for generation in range(max_generation):

        args_list = [
            (i, genome_weights[i], genome_activations[i], seed)
            for i in range(population_size)
        ]

        with Pool(processes=cpu_count()) as pool:
            rewards = pool.map(evaluate_individual, args_list)


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
        state = np.array(state[0]).ravel().reshape(1, -1)
        model = model_ops.build_model(W=genome_weights[np.argmax(rewards)], activations=genome_activations[np.argmax(rewards)], model_type='MLP')
        
        while True:

            action = model_ops.predict_from_memory(state, model, cuda=True)
            
            action = data_ops.normalization(action, dtype=action.dtype)
            action = np.argmax(action)
            next_state, reward, done, truncated, _ = env.step(int(action))
            next_state = np.array(next_state).ravel().reshape(1, -1)
            reward_sum += reward
            state = next_state
            i += 1
            if done or truncated or i == 1000:
                state = env.reset(seed=seed)[0]
                state = np.array(state[0]).ravel().reshape(1, -1)
                break

        model = model_ops.build_model(W=genome_weights[np.argmax(rewards)], activations=genome_activations[np.argmax(rewards)], model_type='MLP')
        model_ops.save_model(model, model_name=f'galaxian_gen{generation}')

        genome_weights, genome_activations = ene.evolver(
            genome_weights, genome_activations, generation,
            np.array(rewards, dtype=np.float32), show_info=True, is_mlp=True
        )

    env.close()


    # --- TESTING THE MODEL ---
    model = model_ops.load_model(model_name=f'galaxian_gen{max_generation-1}', model_path='')

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
    state = np.array(state[0]).ravel().reshape(1, -1)
    reward_sum = 0
    rewards = [0] * 1000
    for i in range(1000):
        while True:
            # Action Calculation
            action = model_ops.predict_from_memory(state, model, cuda=True)
            
            action = data_ops.normalization(action, dtype=action.dtype)
            action = np.argmax(action)
            next_state, reward, done, truncated, _ = env.step(int(action))
            next_state = np.array(next_state).ravel().reshape(1, -1)
            reward_sum += reward
            state = next_state

            if done or truncated:
                state = env.reset(seed=seed)[0]
                state = np.array(state[0]).ravel().reshape(1, -1)
                rewards[i] = reward_sum
                reward_sum = 0
                print(f'return: {rewards[i]}')

                break
        
    print(f'average return: {np.mean(np.array(rewards))}')

if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn')
    main()