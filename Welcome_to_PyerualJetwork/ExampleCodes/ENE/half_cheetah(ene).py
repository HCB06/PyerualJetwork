import gymnasium as gym
import numpy as np
from pyerualjetwork.cpu import activation_functions
from pyerualjetwork import ene, model_ops
import random
from multiprocessing import Pool, cpu_count
import time

# --- ENE WITH PLAN: ---

def evaluate_individual(args):
    individual_idx, W, A, seeds = args
    env = gym.make('HalfCheetah-v4')
    env.model.opt.timestep = 0.003
    total_reward = 0
    model = model_ops.build_model(W=W, activations=A, model_type='PLAN')

    for seed in seeds:
        state = env.reset(seed=seed)[0]
        done = truncated = False
        reward_sum = 0

        while not done and not truncated:
            action = model_ops.predict_from_memory(state, model)
            action = activation_functions.tanh(action)
            state, reward, done, truncated, _ = env.step(action)
            reward_sum += reward
        total_reward += reward_sum

    env.close()
    return total_reward / len(seeds)


def main():
    env_name = 'HalfCheetah-v4'
    env.model.opt.timestep = 0.003
    input_shape = gym.make(env_name).observation_space.shape[0]
    output_shape = gym.make(env_name).action_space.shape[0]
    population_size = 300
    max_generation = 200
    sample_size = 1
    num_threads = cpu_count()
    
    start_time_1 = time.time()

    genome_weights, genome_activations = ene.define_genomes(
        input_shape, output_shape, population_size,
        neurons=[256, 256], activation_functions=['tanh', 'tanh']
    )

    seeds = [random.randint(0, 2**32 - 1) for _ in range(sample_size)]
    max_rewards_1 = []

    for generation in range(max_generation):
        args_list = [
            (i, genome_weights[i], genome_activations[i], seeds)
            for i in range(population_size)
        ]

        with Pool(processes=num_threads) as pool:
            rewards = pool.map(evaluate_individual, args_list)

        best_idx = np.argmax(rewards)
        best_model = model_ops.build_model(
            W=genome_weights[best_idx],
            activations=genome_activations[best_idx],
            model_type='PLAN'
        )

        avg_test_reward = np.max(rewards) / sample_size
        max_rewards_1 = avg_test_reward
    
        # Geçen süreyi kaydet (saniye cinsinden)
        elapsed_time = time.time() - start_time_1
        time_stamps_1 = elapsed_time

        model_ops.save_model(best_model, model_name=f'cheetah_expert_gen{generation}', show_info=False)

        genome_weights, genome_activations = ene.evolver(
            genome_weights, genome_activations, generation,
            np.array(rewards, dtype=np.float32),
            show_info=True, is_mlp=True, bar_status=False, save_best_genome=False
        )

        seeds = [random.randint(0, 2**32 - 1) for _ in range(sample_size)]
        
        # Sadece sonuncu değeri önceki satırları silmeden ekle
        with open("fbn_rewards_5.txt", "a", encoding="utf-8") as file:
            file.write(str(max_rewards_1) + "\n")

        with open("fbn_times_5.txt", "a", encoding="utf-8") as file:
            file.write(str(time_stamps_1) + "\n")

    input("Press Enter to continue...")
    model = model_ops.load_model(model_name=f'cheetah_expert_gen{max_generation - 1}', model_path='')
    print("Model başarıyla yüklendi! 💾")

    env = gym.make(env_name, render_mode='human')
    env.model.opt.timestep = 0.003
    state = env.reset()[0]
    reward_sum = 0
    rewards = []

    for i in range(1, 101):
        while True:
            action = model_ops.predict_from_memory(state, model)
            action = activation_functions.tanh(action)
            state, reward, done, truncated, _ = env.step(action)

            reward_sum += reward

            if done or truncated:
                state = env.reset()[0]
                rewards.append(reward_sum)
                print(f"[Test] Episode {i} reward: {reward_sum:.2f}")
                reward_sum = 0
                break

    env.close()

    print(f"\n✅ Average of 100 episode: {np.mean(np.array(rewards)):.2f}")

if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn')
    main()