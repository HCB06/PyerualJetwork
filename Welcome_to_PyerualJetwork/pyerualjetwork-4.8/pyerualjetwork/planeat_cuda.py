""" 
MAIN MODULE FOR PLANEAT

Examples: https://github.com/HCB06/PyerualJetwork/tree/main/Welcome_to_PyerualJetwork/ExampleCodes

PyerualJetwork document: https://github.com/HCB06/Anaplan/blob/main/Welcome_to_Anaplan/ANAPLAN_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf

@author: Hasan Can Beydili
@YouTube: https://www.youtube.com/@HasanCanBeydili
@Linkedin: https://www.linkedin.com/in/hasan-can-beydili-77a1b9270/
@Instagram: https://www.instagram.com/canbeydilj
@contact: tchasancan@gmail.com
"""

import cupy as cp
import numpy as np
import random
import math
import copy

### LIBRARY IMPORTS ###
from .data_operations_cuda import normalization, non_neg_normalization
from .ui import loading_bars, initialize_loading_bar
from .activation_functions_cuda import apply_activation, all_activations

def define_genomes(input_shape, output_shape, population_size, hidden=0, neurons=None, activation_functions=None, dtype=cp.float32):
   """
   Initializes a population of genomes, where each genome is represented by a set of weights 
   and an associated activation function. Each genome is created with random weights and activation 
   functions are applied and normalized. (Max abs normalization.)

   Args:

      input_shape (int): The number of input features for the neural network.

      output_shape (int): The number of output features for the neural network.
      
      population_size (int): The number of genomes (individuals) in the population.
      
      hidden (int, optional): If you dont want train PLAN model this parameter represents a hidden layer count for MLP model. Default: 0 (PLAN)

      neurons (list[int], optional): If you dont want train PLAN model this parameter represents neuron count of each hidden layer for MLP. Default: None (PLAN)

      activation_functions (list[str], optional): If you dont want train PLAN model this parameter represents activation function of each hidden layer for MLP. Default: None (PLAN) NOTE: THIS EFFECTS HIDDEN LAYERS OUTPUT. NOT OUTPUT LAYER!

      dtype (cupy.dtype): Data type for the arrays. np.float32 by default. Example: cp.float64 or cp.float16.

   Returns:
      tuple: A tuple containing:
         - population_weights (cupy.ndarray): A 2D numpy array of shape (population_size, output_shape, input_shape) representing the 
            weight matrices for each genome.
         - population_activations (list): A list of activation functions applied to each genome.
            
   Notes:
      The weights are initialized randomly within the range [-1, 1]. 
      Activation functions are selected randomly from a predefined list `all_activations()`.
      The weights for each genome are then modified by applying the corresponding activation function 
      and normalized using the `normalization()` function. (Max abs normalization.)
   """
   if hidden > 0:
      population_weights = [[0] * (hidden + 1) for _ in range(population_size)]
      population_activations = [[0] * (hidden) for _ in range(population_size)]
   
      if len(neurons) != hidden:
         raise ValueError('hidden parameter and neurons list length must be equal.')
   

      for i in range(len(population_weights)):

         for l in range(hidden + 1):
            
            if l == 0:
               population_weights[i][l] = cp.random.uniform(-1, 1, (neurons[l], input_shape)).astype(dtype)

            elif l == hidden:
               population_weights[i][l] = cp.random.uniform(-1, 1, (output_shape, neurons[l-1])).astype(dtype)
            
            else:
               population_weights[i][l] = cp.random.uniform(-1, 1, (neurons[l], neurons[l-1])).astype(dtype)
            
            if l != hidden:
               population_activations[i][l] = activation_functions[l]

            # ACTIVATIONS APPLYING IN WEIGHTS SPECIFIC OUTPUT CONNECTIONS (MORE PLAN LIKE FEATURES(FOR NON-LINEARITY)):

            for j in range(population_weights[i][l].shape[0]):

               population_weights[i][l][j,:] = apply_activation(population_weights[i][l][j,:], population_activations[i])
               population_weights[i][l][j,:] = normalization(population_weights[i][l][j,:], dtype=dtype)

      return population_weights, population_activations

   else:
      population_weights = [0] * population_size
      population_activations = [0] * population_size

      except_this = ['spiral', 'circular']
      activations = [item for item in all_activations() if item not in except_this] # SPIRAL AND CIRCULAR ACTIVATION DISCARDED

      for i in range(len(population_weights)):

         population_weights[i] = cp.random.uniform(-1, 1, (output_shape, input_shape)).astype(dtype, copy=False)
         population_activations[i] = activations[int(random.uniform(0, len(activations)-1))]

         # ACTIVATIONS APPLYING IN WEIGHTS SPECIFIC OUTPUT CONNECTIONS (MORE PLAN LIKE FEATURES(FOR NON-LINEARITY)):

         for j in range(population_weights[i].shape[0]):

            population_weights[i][j,:] = apply_activation(population_weights[i][j,:], population_activations[i])
            population_weights[i][j,:] = normalization(population_weights[i][j,:], dtype=dtype)

      return cp.array(population_weights, dtype=dtype), population_activations

def evolver(weights, 
            activation_potentiations, 
            what_gen, 
            fitness,
            weight_evolve=True,
            show_info=False, 
            policy='aggressive', 
            bad_genomes_selection_prob=None, 
            bar_status=True, 
            strategy='normal_selective',
            bad_genomes_mutation_prob=None,  
            fitness_bias=1, 
            cross_over_mode='tpm',
            activation_mutate_add_prob=0.5, 
            activation_mutate_delete_prob=0.5, 
            activation_mutate_change_prob=0.5,  
            activation_selection_add_prob=0.5,
            activation_selection_change_prob=0.5, 
            activation_selection_threshold=20,
            activation_mutate_prob=0.5,
            activation_mutate_threshold=20,
            weight_mutate_threshold=16,
            weight_mutate_prob=1,
            is_mlp=False,
            save_best_genome=False,
            dtype=cp.float32):
   """
   Applies the evolving process of a population of genomes using selection, crossover, mutation, and activation function potentiation.
   The function modifies the population's weights and activation functions based on a specified policy, mutation probabilities, and strategy.

   'selection' args effects cross-over.
   'mutate' args effects mutation.

   Args:
      weights (cupy.ndarray): Array of weights for each genome. 
         (first returned value of define_genomes function)
      
      activation_potentiations (list[str]): A list of activation functions for each genome. 
         (second returned value of define_genomes function) NOTE!: 'activation potentiations' for PLAN 'activation functions' for MLP.
      
      what_gen (int): The current generation number, used for informational purposes or logging.
      
      fitness (cupy.ndarray): A 1D array containing the fitness values of each genome. 
         The array is used to rank the genomes based on their performance. PLANEAT maximizes or minimizes this fitness based on the `target_fitness` parameter.
      
      weight_evolve (bool, optional): Are weights to be evolves or just activation combinations Default: True. Note: Regardless of whether this parameter is True or False, you must give the evolver function a list of weights equal to the number of activation potentiations. You can create completely random weights if you want. If this parameter is False, the weights entering the evolver function and the resulting weights will be exactly the same.

      show_info (bool, optional): If True, prints information about the current generation and the 
         maximum reward obtained. Also shows the current configuration. Default is False.
      
      strategy (str, optional): The strategy for combining the best and bad genomes. Options:
         - 'normal_selective': Normal selection based on reward, where a portion of the bad genes are discarded.
         - 'more_selective': A more selective strategy, where fewer bad genes survive.
         - 'less_selective': A less selective strategy, where more bad genes survive.
         Default is 'normal_selective'.
      
      bar_status (bool, optional): Loading bar status during evolving process of genomes. True or False. Default: True

      policy (str, optional): The selection policy that governs how genomes are selected for reproduction. Options:
            
            - 'aggressive': Aggressive policy using very aggressive selection policy. 
                  Advantages: fast training. 
                  Disadvantages: may lead to fitness stuck in a local maximum or minimum.

            - 'explorer': Explorer policy increases population diversity.
                  Advantages: fitness does not get stuck at local maximum or minimum.
                  Disadvantages: slow training.
                  
                  Suggestions: Use hybrid and dynamic policy. When fitness appears stuck, switch to the 'explorer' policy.

            Default: 'aggressive'.
      
      fitness_bias (float, optional): Fitness bias must be a probability value between 0 and 1 that determines the effect of fitness on the crossover process. Default: 1`.
      
      bad_genomes_mutation_prob (float, optional): The probability of applying mutation to the bad genomes. 
         Must be in the range [0, 1]. Also affects the mutation probability of the best genomes inversely. 
         For example, a value of 0.7 for bad genomes implies 0.3 for best genomes. Default: Determined by `policy`.
      
      bad_genomes_selection_prob (float, optional): The probability of crossover parents are bad genomes ? [0-1] Default: Determined by `policy`.

      activation_mutate_prob (float, optional): The probability of applying mutation to the activation functions. 
         Must be in the range [0, 1]. Default is 0.5 (50%).
      
      cross_over_mode (str, optional): Specifies the crossover method to use. Options:
         - 'tpm': Two-Point Matrix Crossover.
         Default is 'tpm'.
      
      activation_mutate_add_prob (float, optional): The probability of adding a new activation function to the genome for mutation. 
         Must be in the range [0, 1]. Default is 0.5.
      
      activation_mutate_delete_prob (float, optional): The probability of deleting an existing activation function 
         from the genome for mutation. Must be in the range [0, 1]. Default is 0.5.
      
      activation_mutate_change_prob (float, optional): The probability of changing an activation function in the genome for mutation. 
         Must be in the range [0, 1]. Default is 0.5.
      
      weight_mutate_prob (float, optional): The probability of mutating a weight in the genome. 
         Must be in the range [0, 1]. Default is 1 (%100).
      
      weight_mutate_threshold (int): Determines max how much weight mutaiton operation applying. (Function automaticly determines to min) Default: 16

      activation_selection_add_prob (float, optional): The probability of adding an existing activation function for crossover.
         Must be in the range [0, 1]. Default is 0.5. (WARNING! Higher values increase complexity. For faster training, increase this value.)
      
      activation_selection_change_prob (float, optional): The probability of changing an activation function in the genome for crossover. 
         Must be in the range [0, 1]. Default is 0.5.
      
      activation_mutate_threshold (int): Determines max how much activation mutaiton operation applying. (Function automaticly determines to min) Default: 20
      
      activation_selection_threshold (int, optional): Determines max how much activaton transferable to child from undominant parent. (Function automaticly determines to min) Default: 20

      is_mlp (bool, optional): Evolve PLAN model or MLP model ? Default: False (PLAN)

      save_best_genome (bool, optional): Save the best genome of the previous generation to the next generation. Default: False

      dtype (cupy.dtype): Data type for the arrays. Default: cp.float32. 
         Example: cp.float64 or cp.float16 [fp32 for balanced devices, fp64 for strong devices, fp16 for weak devices: not recommended!].

   Raises:
      ValueError: 
         - If `policy` is not one of the specified values ('aggressive', 'explorer').
         - If 'strategy' is not one of the specified values ('less_selective', 'normal_selective', 'more_selective')
         - If `cross_over_mode` is not one of the specified values ('tpm').
         - If `bad_genomes_mutation_prob`, `activation_mutate_prob`, or other probability parameters are not in the range 0 and 1.
         - If the population size is odd (ensuring an even number of genomes is required for proper selection).
         - If 'fitness_bias' value is not in range 0 and 1.

   Returns:
      tuple: A tuple containing:
         - weights (numpy.ndarray): The updated weights for the population after selection, crossover, and mutation. 
                                    The shape is (population_size, output_shape, input_shape).
         - activation_potentiations (list): The updated list of activation functions for the population.

   Notes:
      - **Selection Process**: 
         - The genomes are sorted by their fitness (based on `fitness`), and then split into "best" and "bad" halves. 
         - The best genomes are retained, and the bad genomes are modified based on the selected strategy.
         
      - **Crossover Strategies**:
         - The **'cross_over'** strategy performs crossover, where parts of the best genomes' weights are combined with other good genomes to create new weight matrices.
         
      - **Mutation**:
         - Mutation is applied to both the best and bad genomes, depending on the mutation probability and the `policy`.
         - `bad_genomes_mutation_prob` determines the probability of applying mutations to the bad genomes.
         - If `activation_mutate_prob` is provided, activation function mutations are applied to the genomes based on this probability.
         
      - **Population Size**: The population size must be an even number to properly split the best and bad genomes. If `fitness` has an odd length, an error is raised.
      
      - **Logging**: If `show_info=True`, the current generation and the maximum reward from the population are printed for tracking the learning progress.

   Example:
      ```python
      weights, activation_potentiations = planeat_cuda.evolver(weights, activation_potentiations, 1, fitness, show_info=True, strategy='normal_selective', policy='aggressive')
      ```

      - The function returns the updated weights and activations after processing based on the chosen strategy, policy, and mutation parameters.
"""
    
### ERROR AND CONFIGURATION CHECKS:
   if strategy == 'normal_selective':
      if bad_genomes_mutation_prob is None: bad_genomes_mutation_prob = 0.7 # EFFECTS MUTATION
      if bad_genomes_selection_prob is None: bad_genomes_selection_prob = 0.25 # EFFECTS CROSS-OVER

   elif strategy == 'more_selective':
      if bad_genomes_mutation_prob is None: bad_genomes_mutation_prob = 0.85 # EFFECTS MUTATION
      if bad_genomes_selection_prob is None: bad_genomes_selection_prob = 0.1 # EFFECTS CROSS-OVER

   elif strategy == 'less_selective':
      if bad_genomes_mutation_prob is None: bad_genomes_mutation_prob = 0.6 # EFFECTS MUTATION
      if bad_genomes_selection_prob is None: bad_genomes_selection_prob = 0.5 # EFFECTS CROSS-OVER

   else:
      raise ValueError("strategy parameter must be: 'normal_selective' or 'more_selective' or 'less_selective'")

   if ((activation_mutate_add_prob < 0 or activation_mutate_add_prob > 1) or 
      (activation_mutate_change_prob < 0 or activation_mutate_change_prob > 1) or 
      (activation_mutate_delete_prob < 0 or activation_mutate_delete_prob > 1) or 
      (weight_mutate_prob < 0 or weight_mutate_prob > 1) or 
      (activation_selection_add_prob < 0 or activation_selection_add_prob > 1) or (
      activation_selection_change_prob < 0 or activation_selection_change_prob > 1)):
      
      raise ValueError("All hyperparameters ending with 'prob' must be a number between 0 and 1.")

   if fitness_bias < 0 or fitness_bias > 1: raise ValueError("fitness_bias value must be a number between 0 and 1.")

   if bad_genomes_mutation_prob is not None:
      if bad_genomes_mutation_prob < 0 or bad_genomes_mutation_prob > 1:
         raise ValueError("bad_genomes_mutation_prob parameter must be float and 0-1 range")
      
   if activation_mutate_prob is not None:
      if activation_mutate_prob < 0 or activation_mutate_prob > 1:
         raise ValueError("activation_mutate_prob parameter must be float and 0-1 range")
      
   if len(fitness) % 2 == 0:
      slice_center = int(len(fitness) / 2)

   else:
      raise ValueError("genome population size must be even number. for example: not 99, make 100 or 98.")

   if weight_evolve is False: origin_weights = cp.copy(weights) if is_mlp else copy.deepcopy(weights)

   if is_mlp:
      activation_mutate_add_prob = 0
      activation_selection_add_prob = 0
      activation_mutate_delete_prob = 0

   if isinstance(weights, list):
        
        for i in range(len(weights)):
            
            for j in range(len(weights[i])):

               weights[i][j] = weights[i][j].get()

        weights = np.array(weights, dtype=object)

### FITNESS LIST IS SORTED IN ASCENDING ORDER, AND THE WEIGHT AND ACTIVATIONS OF EACH GENOME ARE SORTED ACCORDING TO THIS ORDER:

   previous_best_genome_index = np.argmax(fitness)

   sort_indices = cp.argsort(fitness)

   fitness = fitness[sort_indices]
   weights = weights[sort_indices]

   activation_potentiations = [activation_potentiations[int(i)] for i in sort_indices]

### GENOMES ARE DIVIDED INTO TWO GROUPS: GOOD GENOMES AND BAD GENOMES:

   good_weights = weights[slice_center:]
   bad_weights = weights[:slice_center]
   best_weight = cp.copy(good_weights[-1])

   good_activations = list(activation_potentiations[slice_center:])
   bad_activations = list(activation_potentiations[:slice_center])
   best_activations = good_activations[-1].copy() if isinstance(good_activations[-1], list) else good_activations[-1]


### PLANEAT IS APPLIED ACCORDING TO THE SPECIFIED POLICY, STRATEGY, AND PROBABILITY CONFIGURATION:
   
   bar_format = loading_bars()[0]
   
   if bar_status: progress = initialize_loading_bar(len(bad_weights), desc="GENERATION: " + str(what_gen), bar_format=bar_format, ncols=50)
   normalized_fitness = non_neg_normalization(fitness, dtype=dtype)

   best_fitness = normalized_fitness[-1]
   epsilon = cp.finfo(float).eps

   child_W = cp.copy(bad_weights)
   child_act = bad_activations.copy()

   mutated_W = cp.copy(bad_weights)
   mutated_act = bad_activations.copy()


   for i in range(len(bad_weights)):
      
      if policy == 'aggressive':
         first_parent_W = best_weight
         first_parent_act = best_activations
         first_parent_fitness = best_fitness

      elif policy == 'explorer':
         first_parent_W = good_weights[i]
         first_parent_act = good_activations[i]
         first_parent_fitness = normalized_fitness[len(good_weights) + i]

      else: raise ValueError("policy parameter must be: 'aggressive' or 'explorer'")
         
      second_parent_W, second_parent_act, s_i = second_parent_selection(good_weights, bad_weights, good_activations, bad_activations, bad_genomes_selection_prob)

      if is_mlp:
            for l in range(len(first_parent_W)):
               
               if l == 0:
                  act_l = 0
               else:
                  act_l = l - 1

               child_W[i][l], child_act[i][act_l] = cross_over(cp.array(first_parent_W[l]),
                                                      cp.array(second_parent_W[l]),
                                                      first_parent_act[act_l],
                                                      second_parent_act[act_l],
                                                      cross_over_mode=cross_over_mode,
                                                      activation_selection_add_prob=activation_selection_add_prob,
                                                      activation_selection_change_prob=activation_selection_change_prob,
                                                      activation_selection_threshold=activation_selection_threshold,
                                                      bad_genomes_selection_prob=bad_genomes_selection_prob,
                                                      first_parent_fitness=first_parent_fitness,
                                                      fitness_bias=fitness_bias,
                                                      second_parent_fitness=normalized_fitness[s_i],
                                                      weight_evolve=weight_evolve,
                                                      epsilon=epsilon
                                                      )
               
               
      else:
            child_W[i], child_act[i] = cross_over(first_parent_W,
                                                   second_parent_W,
                                                   first_parent_act,
                                                   second_parent_act,
                                                   cross_over_mode=cross_over_mode,
                                                   activation_selection_add_prob=activation_selection_add_prob,
                                                   activation_selection_change_prob=activation_selection_change_prob,
                                                   activation_selection_threshold=activation_selection_threshold,
                                                   bad_genomes_selection_prob=bad_genomes_selection_prob,
                                                   first_parent_fitness=first_parent_fitness,
                                                   fitness_bias=fitness_bias,
                                                   second_parent_fitness=normalized_fitness[s_i],
                                                   weight_evolve=weight_evolve,
                                                   epsilon=epsilon
                                                   )
  
      mutation_prob = random.uniform(0, 1)

      if mutation_prob > bad_genomes_mutation_prob:
            genome_W = good_weights[i]
            genome_act = good_activations[i]

            fitness_index = int(len(good_weights) + i)

      else:
            genome_W = bad_weights[i]
            genome_act = bad_activations[i]

            fitness_index = i

      if is_mlp:
         for l in range(len(genome_W)):

            if l == 0:
               act_l = 0
            else:
               act_l = l - 1

            mutated_W[i][l], mutated_act[i][act_l] = mutation(cp.array(genome_W[l]), 
                                                   genome_act[act_l],
                                                   activation_mutate_prob=activation_mutate_prob,
                                                   activation_add_prob=activation_mutate_add_prob,
                                                   activation_delete_prob=activation_mutate_delete_prob, 
                                                   activation_change_prob=activation_mutate_change_prob, 
                                                   weight_mutate_prob=weight_mutate_prob, 
                                                   weight_mutate_threshold=weight_mutate_threshold,
                                                   genome_fitness=normalized_fitness[fitness_index],
                                                   activation_mutate_threshold=activation_mutate_threshold,
                                                   weight_evolve=weight_evolve,
                                                   epsilon=epsilon
                                                   )
      else:
         mutated_W[i], mutated_act[i] = mutation(genome_W, 
                                                genome_act,
                                                activation_mutate_prob=activation_mutate_prob,
                                                activation_add_prob=activation_mutate_add_prob,
                                                activation_delete_prob=activation_mutate_delete_prob, 
                                                activation_change_prob=activation_mutate_change_prob, 
                                                weight_mutate_prob=weight_mutate_prob, 
                                                weight_mutate_threshold=weight_mutate_threshold,
                                                genome_fitness=normalized_fitness[fitness_index],
                                                activation_mutate_threshold=activation_mutate_threshold,
                                                weight_evolve=weight_evolve,
                                                epsilon=epsilon
                                                )
            
      if bar_status: progress.update(1)

   if is_mlp:
      for i in range(len(child_W)):
         for j in range(len(child_W[i])):
            child_W[i][j] = cp.array(child_W[i][j], dtype=dtype)
      
      child_W = list(child_W)
      mutated_W = list(mutated_W)
      weights = child_W + mutated_W
   else:
      weights = cp.vstack((child_W, mutated_W), dtype=dtype)

   activation_potentiations = child_act + mutated_act

   if save_best_genome:
      weights[0] = best_weight
      activation_potentiations[0] = best_activations

   ### INFO PRINTING CONSOLE
   
   if show_info == True:
      print("\nGENERATION:", str(what_gen) + ' FINISHED \n')
      print("*** Configuration Settings ***")
      print("  POPULATION SIZE: ", str(len(weights)))
      print("  STRATEGY: ", strategy)
      print("  CROSS OVER MODE: ", cross_over_mode)
      print("  POLICY: ", policy)
      print("  BAD GENOMES MUTATION PROB: ", str(bad_genomes_mutation_prob))
      print("  GOOD GENOMES MUTATION PROB: ", str(round(1 - bad_genomes_mutation_prob, 2)))
      print("  BAD GENOMES SELECTION PROB: ", str(bad_genomes_selection_prob))
      print("  WEIGHT MUTATE PROB: ", str(weight_mutate_prob))
      print("  WEIGHT MUTATE THRESHOLD: ", str(weight_mutate_threshold))
      print("  ACTIVATION MUTATE PROB: ", str(activation_mutate_prob))
      print("  ACTIVATION MUTATE THRESHOLD: ", str(activation_mutate_threshold))
      print("  ACTIVATION MUTATE ADD PROB: ", str(activation_mutate_add_prob))
      print("  ACTIVATION MUTATE DELETE PROB: ", str(activation_mutate_delete_prob))
      print("  ACTIVATION MUTATE CHANGE PROB: ", str(activation_mutate_change_prob))
      print("  ACTIVATION SELECTION THRESHOLD:", str(activation_selection_threshold))
      print("  ACTIVATION SELECTION ADD PROB: ", str(activation_selection_add_prob))
      print("  ACTIVATION SELECTION CHANGE PROB: ", str(activation_selection_change_prob))
      print("  FITNESS BIAS: ", str(fitness_bias))
      print("  SAVE BEST GENOME: ", str(save_best_genome) + '\n')

      print("*** Performance ***")
      print("  MAX FITNESS: ", str(cp.round(max(fitness), 2)))
      print("  MEAN FITNESS: ", str(cp.round(cp.mean(fitness), 2)))
      print("  MIN FITNESS: ", str(cp.round(min(fitness), 2)) + '\n')

      print("  BEST GENOME ACTIVATION LENGTH: ", str(len(best_activations)))
      print("  PREVIOUS BEST GENOME ACTIVATION LENGTH: ", str(len(best_activations)))
      print("  PREVIOUS BEST GENOME INDEX: ", str(previous_best_genome_index) + '\n')
      
   if weight_evolve is False: weights = origin_weights

   return weights, activation_potentiations


def evaluate(Input, weights, activation_potentiations, is_mlp=False):
   """
    Evaluates the performance of a population of genomes, applying different activation functions 
    and weights depending on whether reinforcement learning mode is enabled or not.

    Args:
        Input (list or cupy.ndarray): A list or 2D numpy or cupy array where each element represents
                                               a genome (A list of input features for each genome, or a single set of input features for one genome).
        
        weights (list or cupy.ndarray): A list or 2D numpy array of weights corresponding to each genome 
                                         in `x_population`. This determines the strength of connections.
        
        activation_potentiations (list or str): A list where each entry represents an activation function 
                                                or a potentiation strategy applied to each genome. If only one 
                                                activation function is used, this can be a single string.
        is_mlp (bool, optional): Evaluate PLAN model or MLP model ? Default: False (PLAN)
    
    Returns:
        list: A list of outputs corresponding to each genome in the population after applying the respective 
              activation function and weights.

    Example:
        ```python
        outputs = evaluate(Input, weights, activation_potentiations)
        ```

    - The function returns a list of outputs after processing the population, where each element corresponds to 
      the output for each genome in population.
   """
   ### THE OUTPUTS ARE RETURNED WHERE EACH GENOME'S OUTPUT MATCHES ITS INDEX:
   
   if isinstance(activation_potentiations, str):
    activation_potentiations = [activation_potentiations]
   else:
    activation_potentiations = [item if isinstance(item, list) else [item] for item in activation_potentiations]

   if is_mlp:
      
      layer = Input
      for i in range(len(weights)):
         if i != len(weights) - 1: layer = apply_activation(layer, activation_potentiations[i])
         layer = layer @ weights[i].T
      
      return layer

   else:

      Input = apply_activation(Input, activation_potentiations)
      result = Input @ weights.T

      return result

def cross_over(first_parent_W,
               second_parent_W,
               first_parent_act,
               second_parent_act,
               cross_over_mode,
               activation_selection_add_prob,
               activation_selection_change_prob,
               activation_selection_threshold,
               bad_genomes_selection_prob,
               first_parent_fitness,
               second_parent_fitness,
               fitness_bias,
               weight_evolve,
               epsilon):
   """
    Performs a crossover operation on two sets of weights and activation functions.
    This function combines two individuals (represented by their weights and activation functions) 
    to create a new individual by exchanging parts of their weight matrices and activation functions.

    Args:
        first_parent_W (cupy.ndarray): The weight matrix of the first individual (parent).
        
        second_parent_W (numpy.ndarray): The weight matrix of the second individual (parent).
        
        first_parent_act (str or list): The activation function(s) of the first individual.
        
        second_parent_act (str or list): The activation function(s) of the second individual.
        
        cross_over_mode (str): Determines the crossover method to be used. Options:
            - 'tpm': Two-Point Matrix Crossover, where sub-matrices of weights are swapped between parents.
        
        activation_selection_add_prob (float): Probability of adding new activation functions 
            from the second parent to the child genome.
        
        activation_selection_change_prob (float): Probability of replacing an activation function in the child genome 
            with one from the second parent.
        
        activation_selection_threshold (int): Determines max how much activaton transferable to child from undominant parent. (Function automaticly determines to min)
        
        bad_genomes_selection_prob (float): Probability of selecting a "bad" genome for replacement with the offspring.
        
        first_parent_fitness (float): Fitness score of the first parent.
        
        second_parent_fitness (float): Fitness score of the second parent.
        
        fitness_bias (float): A bias factor used to favor fitter parents during crossover operations.

        weight_evolve (bool, optional): Are weights to be evolves or just activation combinations.

        epsilon (float): Small epsilon constant

    Returns:
        tuple: A tuple containing:
            - child_W (numpy.ndarray): The weight matrix of the new individual created by crossover.
            - child_act (list): The list of activation functions of the new individual created by crossover.

    Notes:
        - The crossover is performed based on the selected `cross_over_mode`.
        - In 'tpm' mode, random sub-matrices from the parent weight matrices are swapped.
        - Activation functions from both parents are combined using the probabilities and rates provided.

    Example:
        ```python
        new_weights, new_activations = cross_over(
            first_parent_W=parent1_weights, 
            second_parent_W=parent2_weights, 
            first_parent_act=parent1_activations, 
            second_parent_act=parent2_activations, 
            cross_over_mode='tpm',
            activation_selection_add_prob=0.8,
            activation_selection_change_prob=0.5,
            activation_selection_threshold=2,
            bad_genomes_selection_prob=0.7,
            first_parent_fitness=0.9,
            second_parent_fitness=0.85,
            fitness_bias=0.6,
            weight_evolve=True,
            epsilon=cp.finfo(float).eps
        )
        ```
    """
   
  ### THE GIVEN GENOMES' WEIGHTS ARE RANDOMLY SELECTED AND COMBINED OVER A RANDOM RANGE. SIMILARLY, THEIR ACTIVATIONS ARE COMBINED. A NEW GENOME IS RETURNED WITH THE COMBINED WEIGHTS FIRST, FOLLOWED BY THE ACTIVATIONS:
   
   start = 0
   
   row_end = first_parent_W.shape[0]
   col_end = first_parent_W.shape[1]

   total_gene = row_end * col_end
   half_of_gene = int(total_gene / 2)

   decision = dominant_parent_selection(bad_genomes_selection_prob)

   if decision == 'first_parent':
      dominant_parent_W = cp.copy(first_parent_W)
      dominant_parent_act = first_parent_act

      undominant_parent_W = cp.copy(second_parent_W)
      undominant_parent_act = second_parent_act
      succes = second_parent_fitness + epsilon

   elif decision == 'second_parent':
      dominant_parent_W = cp.copy(second_parent_W)
      dominant_parent_act = second_parent_act

      undominant_parent_W = cp.copy(first_parent_W)
      undominant_parent_act = first_parent_act
      succes = first_parent_fitness + epsilon

   if weight_evolve is True:
      while True:

         row_cut_start = int(random.uniform(start, row_end))
         col_cut_start = int(random.uniform(start, col_end))

         row_cut_end = int(random.uniform(start, row_end))
         col_cut_end = int(random.uniform(start, col_end))

         if ((row_cut_end > row_cut_start) and
            (col_cut_end > col_cut_start) and
            (((row_cut_end + 1) - (row_cut_start + 1) * 2) + ((col_cut_end + 1) - (col_cut_start + 1) * 2) <= half_of_gene)):
            break
         
         selection_bias = random.uniform(0, 1)

         if fitness_bias > selection_bias:
            row_cut_start = math.floor(row_cut_start * succes)
            row_cut_end = math.ceil(row_cut_end * succes)

            col_cut_start = math.floor(col_cut_start * succes)
            col_cut_end = math.ceil(col_cut_end * succes)

      child_W = dominant_parent_W

      if cross_over_mode == 'tpm':
         child_W[row_cut_start:row_cut_end, col_cut_start:col_cut_end] = undominant_parent_W[row_cut_start:row_cut_end, col_cut_start:col_cut_end]
   
   else: child_W = dominant_parent_W

   if isinstance(dominant_parent_act, str): dominant_parent_act = [dominant_parent_act]
   if isinstance(undominant_parent_act, str): undominant_parent_act = [undominant_parent_act]

   child_act = list(np.copy(dominant_parent_act))

   activation_selection_add_prob = 1 - activation_selection_add_prob # if prob 0.8 (%80) then 1 - 0.8. Because 0-1 random number probably greater than 0.2
   potential_activation_selection_add = random.uniform(0, 1)

   if potential_activation_selection_add > activation_selection_add_prob:
      
      threshold = abs(activation_selection_threshold / succes)
      new_threshold = threshold
      
      while True:
         
         random_index = int(random.uniform(0, len(undominant_parent_act)))
         random_undominant_activation = undominant_parent_act[random_index]

         child_act.append(random_undominant_activation)
         new_threshold += threshold

         if len(dominant_parent_act) > new_threshold:
            pass
         
         else:
            break

   activation_selection_change_prob = 1 - activation_selection_change_prob
   potential_activation_selection_change_prob = random.uniform(0, 1)

   if potential_activation_selection_change_prob > activation_selection_change_prob:
      
      threshold = abs(activation_selection_threshold / succes)
      new_threshold = threshold
      
      while True:
         
         random_index_undominant = int(random.uniform(0, len(undominant_parent_act)))
         random_index_dominant = int(random.uniform(0, len(dominant_parent_act)))
         random_undominant_activation = undominant_parent_act[random_index_undominant]

         child_act[random_index_dominant] = random_undominant_activation
         new_threshold += threshold

         if len(dominant_parent_act) > new_threshold:
            pass
         
         else:
            break

   return child_W, child_act


def mutation(weight, 
             activations, 
             activation_mutate_prob, 
             activation_add_prob, 
             activation_delete_prob, 
             activation_change_prob, 
             weight_mutate_prob, 
             weight_mutate_threshold, 
             genome_fitness,
             activation_mutate_threshold,
             weight_evolve,
             epsilon):
   """
    Performs mutation on the given weight matrix and activation functions.
    - The weight matrix is mutated by randomly changing its values based on the mutation probability.
    - The activation functions are mutated by adding, removing, or replacing them with predefined probabilities.

    Args:
        weight (cupy.ndarray): The weight matrix to mutate.
        
        activations (list): The list of activation functions to mutate.
        
        activation_mutate_prob (float): The overall probability of mutating activation functions.
        
        activation_add_prob (float): Probability of adding a new activation function.
        
        activation_delete_prob (float): Probability of removing an existing activation function.
        
        activation_change_prob (float): Probability of replacing an existing activation function with a new one.
        
        weight_mutate_prob (float): The probability of mutating weight matrix.
        
        weight_mutate_threshold (float): Determines max how much weight mutaiton operation applying. (Function automaticly determines to min)
        
        genome_fitness (float): Fitness value of genome
    
        activation_mutate_threshold (float): Determines max how much activation mutaiton operation applying. (Function automaticly determines to min) 

        weight_evolve (bool, optional): Are weights to be mutates or just activation combinations.

        epsilon (float): Small epsilon constant

    Returns:
        tuple: A tuple containing:
            - mutated_weight (numpy.ndarray): The weight matrix after mutation.
            - mutated_activations (list): The list of activation functions after mutation.

    Notes:
        - Weight mutation:
            - Each weight has a chance defined by `weight_mutate_prob` to be altered by adding a random value 
              within the range of [0, 1].
        - Activation mutation:
            - If `activation_mutate_prob` is triggered, one or more activation functions can be added, removed, 
              or replaced based on their respective probabilities (`activation_add_prob`, `activation_delete_prob`, 
              `activation_change_prob`).
        - The mutation probabilities should be chosen carefully to balance exploration and exploitation during 
          the optimization process.
   """

   if isinstance(activations, str): activations = [activations]

   if weight_evolve is True:

      weight_mutate_prob = 1 - weight_mutate_prob # if prob 0.8 (%80) then 1 - 0.8. Because 0-1 random number probably greater than 0.2
      potential_weight_mutation = random.uniform(0, 1)

      if potential_weight_mutation > weight_mutate_prob:

            row_end, col_end = weight.shape
            max_threshold = row_end * col_end
            threshold = weight_mutate_threshold / (genome_fitness + epsilon)
   
            n_mutations = min(int(threshold), max_threshold)
                     
            row_indices = cp.random.randint(0, row_end, size=n_mutations)
            col_indices = cp.random.randint(0, col_end, size=n_mutations)
            
            new_values = cp.random.uniform(-1, 1, size=n_mutations)
            
            weight[row_indices, col_indices] = new_values

   activation_mutate_prob = 1 - activation_mutate_prob
   potential_activation_mutation = random.uniform(0, 1)

   if potential_activation_mutation > activation_mutate_prob:
      
      genome_fitness += epsilon
      threshold = abs(activation_mutate_threshold / genome_fitness)
      max_threshold = len(activations)

      new_threshold = threshold
      
      except_this = ['spiral', 'circular']
      all_acts = [item for item in all_activations() if item not in except_this] # SPIRAL AND CIRCULAR ACTIVATION DISCARDED
      
      activation_add_prob = 1 - activation_add_prob
      activation_delete_prob = 1 - activation_delete_prob
      activation_change_prob = 1 - activation_change_prob

      for _ in range(max_threshold):

         potential_activation_add_prob = random.uniform(0, 1)
         potential_activation_delete_prob = random.uniform(0, 1)
         potential_activation_change_prob = random.uniform(0, 1)

         
         if potential_activation_delete_prob > activation_delete_prob and len(activations) > 1:
            
            random_index = random.randint(0, len(activations))
            activations.pop(random_index)
            

         if potential_activation_add_prob > activation_add_prob:

            try:
               
               random_index_all_act = int(random.uniform(0, len(all_acts)))
               activations.append(all_acts[random_index_all_act])

            except:

               activation = activations
               activations = []

               activations.append(activation)
               activations.append(all_acts[int(random.uniform(0, len(all_acts)))])
            
         if potential_activation_change_prob > activation_change_prob:
            
            random_index_all_act = int(random.uniform(0, len(all_acts)))
            random_index_genom_act = int(random.uniform(0, len(activations)))

            activations[random_index_genom_act] = all_acts[random_index_all_act]

         new_threshold += threshold

         if max_threshold > new_threshold: pass
         else: break

   return weight, activations


def second_parent_selection(good_weights, bad_weights, good_activations, bad_activations, bad_genomes_selection_prob):
   
   selection_prob = random.uniform(0, 1)
   random_index = int(random.uniform(0, len(good_weights)))

   if selection_prob > bad_genomes_selection_prob:
      second_selected_W = good_weights[random_index]
      second_selected_act = good_activations[random_index]

   else:
      second_selected_W = bad_weights[random_index]
      second_selected_act = bad_activations[random_index]

   return second_selected_W, second_selected_act, random_index
      

def dominant_parent_selection(bad_genomes_selection_prob):

   selection_prob = random.uniform(0, 1)

   if selection_prob > bad_genomes_selection_prob: decision = 'first_parent'
   else: decision = 'second_parent'

   return decision