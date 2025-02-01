import multiprocessing as mp
from functools import partial
import random
from .planeat import mutation, cross_over, second_parent_selection

def run_process(policy, best_weight, best_activations, good_weights, 
                good_activations, bad_weights, bad_activations, best_fitness, 
                normalized_fitness, child_W, child_act, mutated_W, mutated_act,
                cross_over_mode, activation_selection_add_prob, 
                activation_selection_change_prob, activation_selection_threshold, 
                bad_genomes_selection_prob, fitness_bias, epsilon, 
                bad_genomes_mutation_prob, activation_mutate_prob, 
                activation_mutate_add_prob, activation_mutate_delete_prob, 
                activation_mutate_change_prob, weight_mutate_prob, 
                weight_mutate_threshold, activation_mutate_threshold):
    
    process_func = partial(process_single, policy=policy, best_weight=best_weight, 
                           best_activations=best_activations, good_weights=good_weights, 
                           good_activations=good_activations, bad_weights=bad_weights, 
                           bad_activations=bad_activations, best_fitness=best_fitness, 
                           normalized_fitness=normalized_fitness, child_W=child_W, 
                           child_act=child_act, mutated_W=mutated_W, mutated_act=mutated_act,
                           cross_over_mode=cross_over_mode,
                           activation_selection_add_prob=activation_selection_add_prob,
                           activation_selection_change_prob=activation_selection_change_prob,
                           activation_selection_threshold=activation_selection_threshold,
                           bad_genomes_selection_prob=bad_genomes_selection_prob,
                           fitness_bias=fitness_bias,
                           epsilon=epsilon,
                           bad_genomes_mutation_prob=bad_genomes_mutation_prob,
                           activation_mutate_prob=activation_mutate_prob,
                           activation_mutate_add_prob=activation_mutate_add_prob,
                           activation_mutate_delete_prob=activation_mutate_delete_prob,
                           activation_mutate_change_prob=activation_mutate_change_prob,
                           weight_mutate_prob=weight_mutate_prob,
                           weight_mutate_threshold=weight_mutate_threshold,
                           activation_mutate_threshold=activation_mutate_threshold)

    with mp.Pool() as pool:
        results = pool.map(process_func, range(len(bad_weights)))

    for i, new_child_W, new_child_act, new_mutated_W, new_mutated_act in results:
        child_W[i] = new_child_W
        child_act[i] = new_child_act
        mutated_W[i] = new_mutated_W
        mutated_act[i] = new_mutated_act


    return child_W, child_act, mutated_W, mutated_act


def process_single(i, policy, best_weight, best_activations, 
                   good_weights, good_activations, 
                   bad_weights, bad_activations, 
                   best_fitness, normalized_fitness, 
                   cross_over_mode, 
                   activation_selection_add_prob, activation_selection_change_prob,
                   activation_selection_threshold, 
                   bad_genomes_selection_prob, fitness_bias,
                   epsilon, bad_genomes_mutation_prob, 
                   activation_mutate_prob, activation_mutate_add_prob,
                   activation_mutate_delete_prob, activation_mutate_change_prob,
                   weight_mutate_prob, weight_mutate_threshold,
                   activation_mutate_threshold):

    if policy == 'aggressive':
        first_parent_W = best_weight
        first_parent_act = best_activations
    elif policy == 'explorer':
        first_parent_W = good_weights[i]
        first_parent_act = good_activations[i]
    else:
        raise ValueError("policy parameter must be: 'aggressive' or 'explorer'")
        
    second_parent_W, second_parent_act, s_i = second_parent_selection(
        good_weights, bad_weights, good_activations, bad_activations, bad_genomes_selection_prob)

    new_child_W, new_child_act = cross_over(
        first_parent_W,
        second_parent_W,
        first_parent_act,
        second_parent_act,
        cross_over_mode=cross_over_mode,
        activation_selection_add_prob=activation_selection_add_prob,
        activation_selection_change_prob=activation_selection_change_prob,
        activation_selection_threshold=activation_selection_threshold,
        bad_genomes_selection_prob=bad_genomes_selection_prob,
        first_parent_fitness=best_fitness,
        fitness_bias=fitness_bias,
        second_parent_fitness=normalized_fitness[s_i],
        epsilon=epsilon
    )

    mutation_prob = random.uniform(0, 1)
    if mutation_prob > bad_genomes_mutation_prob:
        genome_W = good_weights[i]
        genome_act = good_activations[i]
        fitness_index = int(len(bad_weights) / 2 + i)
    else:
        genome_W = bad_weights[i]
        genome_act = bad_activations[i]
        fitness_index = i

    new_mutated_W, new_mutated_act = mutation(
        genome_W,
        genome_act,
        activation_mutate_prob=activation_mutate_prob,
        activation_add_prob=activation_mutate_add_prob,
        activation_delete_prob=activation_mutate_delete_prob,
        activation_change_prob=activation_mutate_change_prob,
        weight_mutate_prob=weight_mutate_prob,
        weight_mutate_threshold=weight_mutate_threshold,
        genome_fitness=normalized_fitness[fitness_index],
        activation_mutate_threshold=activation_mutate_threshold,
        epsilon=epsilon
    )
    
    return (i, new_child_W, new_child_act, new_mutated_W, new_mutated_act)