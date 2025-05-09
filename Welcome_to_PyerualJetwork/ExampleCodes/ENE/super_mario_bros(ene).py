from pyerualjetwork import ene, model_ops
import numpy as np
import cv2
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import time  # Süre ölçmek için

# Ortam oluşturma
env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True)
env = JoypadSpace(env, [['right'], ['right', 'A']])

genome_weights, genome_activations = ene.define_genomes(input_shape=7056, output_shape=2, population_size=100, neurons=[256, 256], activation_functions=['tanh', 'tanh'])

genome_weights = np.array(genome_weights)

generation = 0
rewards = [0] * 100

while True:
    for i in range(100):
        # Ajan başına sıfırlama
        state = env.reset()
        state = np.array(state[0])
        reward_sum = 0

        # Zaman sınırını başlat
        start_time = time.time()
        time_limit = 10  # Saniye cinsinden süre sınırı

        while True:
            # Görüntüyü gri tona çevir ve yeniden boyutlandır
            state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
            state = cv2.resize(state, (84, 84))
            state = state / 255.0

            # Aksiyon hesaplama
            model = model_ops.build_model(W=genome_weights[i], activations=genome_activations[i], model_type='MLP')
            output = model_ops.predict_from_memory(np.array(state.flatten()), model)

            action = np.argmax(output)

            # Adım at ve sonucu al
            state, reward, done, truncated, info = env.step(action)
            reward_sum += reward

            # Zaman kontrolü veya diğer koşullarda sıfırla
            elapsed_time = time.time() - start_time
            if done or truncated or elapsed_time > time_limit:
                print(f"\rAgent {i} completed after {elapsed_time:.2f} seconds.", end="")
            
                break
            

        rewards[i] = reward_sum


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
        if isinstance(genome_activations[i], str):
                genome_activations[i] = [genome_activations[i]]
                
        model_ops.save_model(model_name=f'mario{i}', model_path='', W=genome_weights[i], activation_potentiation=genome_activations[i], show_info=False)

    if generation > 20:
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
            
