from pyerualjetwork import ene, model_ops
import numpy as np
import cv2
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import time  # Süre ölçmek için
from multiprocessing import Pool, cpu_count
import random

# --- ENE WITH MLP: ---

def evaluate_individual(args):
    individual_idx, W, A, seeds = args
    env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True)
    env = JoypadSpace(env, [['right'], ['right', 'A']])
    total_reward = 0
    model = model_ops.build_model(W=W, activations=A, model_type='MLP')

    for seed in seeds:
        state = env.reset(seed=seed)[0]
        done = truncated = False
        reward_sum = 0

        while not done and not truncated:
                # Görüntüyü gri tona çevir ve yeniden boyutlandır
                state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
                state = cv2.resize(state, (84, 84))
                state = state / 255.0

                output = model_ops.predict_from_memory(np.array(state.flatten()), model)

                action = np.argmax(output)

                # Adım at ve sonucu al
                state, reward, done, truncated, info = env.step(action)
                reward_sum += reward
        total_reward += reward_sum


    env.close()
    return total_reward / len(seeds)

def main():
    # Ortam oluşturma
    env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True)
    env = JoypadSpace(env, [['right'], ['right', 'A']])

    genome_weights, genome_activations = ene.define_genomes(input_shape=7056, output_shape=2, population_size=100, neurons=[256, 256], activation_functions=['tanh', 'tanh'])

    genome_weights = np.array(genome_weights)

    generation = 0
    rewards = [0] * 100
    seeds = [random.randint(0, 2**32 - 1) for _ in range(1)]

    while True:

        args_list = [
            (i, genome_weights[i], genome_activations[i], seeds)
            for i in range(100)
        ]

        with Pool(processes=cpu_count()) as pool:
            rewards = pool.map(evaluate_individual, args_list)

        best_idx = np.argmax(rewards)
        best_model = model_ops.build_model(
            W=genome_weights[best_idx],
            activations=genome_activations[best_idx],
            model_type='MLP'
        )

        
        env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode='human')
        env = JoypadSpace(env, [['right'], ['right', 'A']])
        total_reward = 0
        model = model_ops.build_model(W=genome_weights[best_idx], activations=genome_activations[best_idx], model_type='MLP')

        for seed in seeds:
            state = env.reset(seed=seed)[0]
            done = truncated = False
            reward_sum = 0

            while not done and not truncated:
                    # Görüntüyü gri tona çevir ve yeniden boyutlandır
                    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
                    state = cv2.resize(state, (84, 84))
                    state = state / 255.0

                    output = model_ops.predict_from_memory(np.array(state.flatten()), model)

                    action = np.argmax(output)

                    # Adım at ve sonucu al
                    state, reward, done, truncated, info = env.step(action)
                    reward_sum += reward
            total_reward += reward_sum
        env.close()

        # Jenerasyon ve genom güncellemesi
        generation += 1
        genome_weights, genome_activations = ene.evolver(
            genome_weights,
            genome_activations,
            generation,
            strategy='more_selective',
            policy='aggressive',
            fitness=np.array(rewards),
            show_info=True
        )
        
        
        for i in range(100):
            model_ops.save_model(model_name=f'mario{i}', model_path='', W=genome_weights[i], activation_potentiation=genome_activations[i], show_info=False)

        if generation > 100:
            break

    env.close()

    model = model_ops.load_model(model_name='mario0', model_path='')

    # Ortam oluşturma
    env = gym_super_mario_bros.make('SuperMarioBros-v0', render_mode='human', apply_api_compatibility=True)
    env = JoypadSpace(env, [['right'], ['right', 'A']])


    while True:
        while True:
            # Ajan başına sıfırlama
            state = env.reset()
            state = np.array(state[0])

            # Zaman sınırını başlat
            start_time = time.time()
            time_limit = 10  # Saniye cinsinden süre sınırı

            while True:
                # Görüntüyü gri tona çevir ve yeniden boyutlandır
                state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
                state = cv2.resize(state, (84, 84))
                state = state / 255.0

                # Aksiyon hesaplama
                output = model_ops.predict_from_memory(np.array(state.flatten()), model)

                action = np.argmax(output)

                # Adım at ve sonucu al
                state, reward, done, truncated, info = env.step(action)
                # Zaman kontrolü veya diğer koşullarda sıfırla
                elapsed_time = time.time() - start_time
                if done or truncated:
                
                    break
                
if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn')
    main()