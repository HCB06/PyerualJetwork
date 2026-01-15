import gymnasium as gym
import numpy as np
from pyerualjetwork.cpu import activation_functions, visualizations
from pyerualjetwork import ene, model_ops
import random
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import copy

# --- evaluate_individual ---
def evaluate_individual(args):
    individual_idx, W, A, seeds, show_visual = args
    env = gym.make('LunarLanderContinuous-v3', render_mode='rgb_array')
    total_reward = 0
    model = model_ops.build_model(W=W, activations=A, model_type='MLP')

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

# --- MAIN LOOP ---
def main():
    env_name = 'LunarLanderContinuous-v3'
    input_shape = gym.make(env_name).observation_space.shape[0]
    output_shape = gym.make(env_name).action_space.shape[0]
    population_size = 300
    max_generation = 200
    sample_size = 25
    num_threads = cpu_count()

    genome_weights, genome_activations = ene.define_genomes(
        input_shape, output_shape, population_size,
        neurons=[8,8], activation_functions=['tanh','tanh']
    )

    seeds = [random.randint(0, 2**32 - 1) for _ in range(sample_size)]
    W_prev = [np.zeros_like(W) for W in genome_weights[0]]
    edge_colors_fixed = None

    plt.ion()
    fig, (ax_land, ax_neuron) = plt.subplots(1, 2, figsize=(16,6))

    for generation in range(max_generation):
        show_visual_trigger = False
        args_list = [
            (i, genome_weights[i], genome_activations[i], seeds, show_visual_trigger)
            for i in range(population_size)
        ]

        with Pool(processes=num_threads) as pool:
            rewards = pool.map(evaluate_individual, args_list)

        best_idx = np.argmax(rewards)
        best_W_new = genome_weights[best_idx]
        best_activations = genome_activations[best_idx]

        # --- Ağırlık geçişi ve edge gradient ---
        state_example = gym.make(env_name).reset()[0]
        _, activations_list = model_ops.predict_from_memory(
            state_example,
            model_ops.build_model(W=best_W_new, activations=best_activations, model_type='MLP'),
            return_activations=True
        )
        edge_colors_fixed = visualizations.draw_neural_web_dynamic(W_prev, best_W_new, activations_list, ax=ax_neuron, steps=15, edge_colors_fixed=edge_colors_fixed)

        # --- En iyi bireyi görselleştir ---
        env = gym.make(env_name, render_mode='rgb_array')
        model = model_ops.build_model(W=best_W_new, activations=best_activations, model_type='MLP')

        state = env.reset()[0]
        done = truncated = False
        while not done and not truncated:
            action, activations_list = model_ops.predict_from_memory(state, model, return_activations=True)
            action = activation_functions.tanh(action)

            frame = env.render()
            ax_land.clear()
            ax_land.imshow(frame)
            ax_land.axis('off')
            ax_land.set_title("Lunar Lander")

            visualizations.draw_neural_web_dynamic(activations_list=activations_list, ax=ax_neuron, steps=1, edge_colors_fixed=edge_colors_fixed)

            state, reward, done, truncated, _ = env.step(action)

        env.close()
        W_prev = copy.deepcopy(best_W_new)

        genome_weights, genome_activations = ene.evolver(
            genome_weights, genome_activations, generation,
            np.array(rewards, dtype=np.float32),
            show_info=True, is_mlp=True, bar_status=False, save_best_genome=False
        )

        rewards = None

if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn')
    main()
