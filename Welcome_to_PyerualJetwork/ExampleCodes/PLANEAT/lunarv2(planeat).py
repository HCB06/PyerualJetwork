import gym
from pyerualjetwork import planeat
import numpy as np

"""
pip install gym
pip install box2d-py
pip install pygame

Note: Before install 'box2d-py' you need to install this stuff (for Windows users):

1 - download and unzip this file: https://sourceforge.net/projects/swig/files/swigwin/swigwin-4.3.0/swigwin-4.3.0.zip/download?use_mirror=netix
1.1 - Add your swig file path to PATH directory of your environment variables.

2 - download and install this: https://visualstudio.microsoft.com/tr/visual-cpp-build-tools/
2.1 - Select C++ build tools option and make sure this options are selected:

a.) MSVC v14.3x (or above) C++ Compiler
b.) Windows 10 SDK (or above)
c.) C++ CMake tools

"""

# Genomlar
genome_weights, genome_activations = planeat.define_genomes(input_shape=8, output_shape=4, population_size=300)
generation = 0
rewards = [0] * 300

reward_sum = 0

while True:
    for i in range(300):

        if i == 0:
            env = gym.make('LunarLander-v2', render_mode='human')
            state = env.reset(seed=75)
            state = np.array(state[0])
        else:
            env.close()
            env = gym.make('LunarLander-v2')
            state = env.reset(seed=75)
            state = np.array(state[0])

        while True:
            
            # Aksiyon hesaplama
            output = planeat.evaluate(
                Input=np.array(state),
                weights=genome_weights[i],
                rl_mode=True,
                activation_potentiations=genome_activations[i]
            )
            action = np.argmax(output)
            state, reward, done, truncated, _ = env.step(action)
            
            reward_sum += reward
            
            if done or truncated:
                state = env.reset(seed=75)
                state = np.array(state[0])
                rewards[i] = reward_sum
                reward_sum = 0
                
                break

    # Jenerasyon ve genom güncellemesi
    generation += 1
    
    genome_weights, genome_activations = planeat.evolver(
        genome_weights,
        genome_activations,
        generation,
        strategy='normal_selective',
        policy='aggressive',
        fitness=np.array(rewards),
        show_info=True,
    )
    
    if generation == 20:
        
        break
