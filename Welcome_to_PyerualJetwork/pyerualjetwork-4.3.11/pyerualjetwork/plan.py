# -*- coding: utf-8 -*-
"""

MAIN MODULE FOR PLAN

Examples: https://github.com/HCB06/PyerualJetwork/tree/main/Welcome_to_PyerualJetwork/ExampleCodes

PLAN document: https://github.com/HCB06/PyerualJetwork/blob/main/Welcome_to_PLAN/PLAN.pdf
PyerualJetwork document: https://github.com/HCB06/PyerualJetwork/blob/main/Welcome_to_PyerualJetwork/PYERUALJETWORK_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf

@author: Hasan Can Beydili
@YouTube: https://www.youtube.com/@HasanCanBeydili
@Linkedin: https://www.linkedin.com/in/hasan-can-beydili-77a1b9270/
@Instagram: https://www.instagram.com/canbeydilj/
@contact: tchasancan@gmail.com
"""

import numpy as np

### LIBRARY IMPORTS ###
from .ui import loading_bars, initialize_loading_bar
from .data_operations import normalization, batcher
from .activation_functions import apply_activation, all_activations
from .model_operations import get_acc, get_preds_softmax
from .memory_operations import optimize_labels
from .loss_functions import categorical_crossentropy, binary_crossentropy
from .fitness_functions import wals
from .visualizations import (
    draw_neural_web,
    display_visualizations_for_learner,
    update_history_plots_for_learner,
    initialize_visualization_for_learner,
    update_neuron_history_for_learner
)

### GLOBAL VARIABLES ###
bar_format_normal = loading_bars()[0]
bar_format_learner = loading_bars()[1]

# BUILD -----

def fit(
    x_train,
    y_train,
    activation_potentiation=['linear'],
    W=None,
    auto_normalization=False,
    dtype=np.float32
):
    """
    Creates a model to fitting data.
    
    fit Args:

        x_train (aray-like[num]): List or numarray of input data.

        y_train (aray-like[num]): List or numarray of target labels. (one hot encoded)

        activation_potentiation (list): For deeper PLAN networks, activation function parameters. For more information please run this code: plan.activations_list() default: [None] (optional)

        W (numpy.ndarray): If you want to re-continue or update model

        auto_normalization (bool, optional): Normalization may solves overflow problem. Default: False

        dtype (numpy.dtype): Data type for the arrays. np.float32 by default. Example: np.float64 or np.float16. [fp32 for balanced devices, fp64 for strong devices, fp16 for weak devices: not reccomended!] (optional)

    Returns:
        numpyarray: (Weight matrix).
    """

    # Pre-check
    
    if len(x_train) != len(y_train): raise ValueError("x_train and y_train must have the same length.")

    weight = np.zeros((len(y_train[0]), len(x_train[0].ravel()))).astype(dtype, copy=False) if W is None else W

    if auto_normalization is True: x_train = normalization(apply_activation(x_train, activation_potentiation))
    elif auto_normalization is False: x_train = apply_activation(x_train, activation_potentiation)
    else: raise ValueError('normalization parameter only be True or False')
    
    weight += y_train.T @ x_train

    return normalization(weight, dtype=dtype)


def learner(x_train, y_train, optimizer, fit_start=True, gen=None, batch_size=1, pop_size=None,
           neural_web_history=False, show_current_activations=False, auto_normalization=False,
           neurons_history=False, early_stop=False, show_history=False, target_loss=None,
           interval=33.33, target_acc=None, loss='categorical_crossentropy', acc_impact=0.9, loss_impact=0.1,
           start_this_act=None, start_this_W=None, dtype=np.float32):
    """
    Optimizes the activation functions for a neural network by leveraging train data to find 
    the most accurate combination of activation potentiation for the given dataset using genetic algorithm NEAT (Neuroevolution of Augmenting Topologies). But modifided for PLAN version. Created by me: PLANEAT. 
    
    Why genetic optimization and not backpropagation?
    Because PLAN is different from other neural network architectures. In PLAN, the learnable parameters are not the weights; instead, the learnable parameters are the activation functions.
    Since activation functions are not differentiable, we cannot use gradient descent or backpropagation. However, I developed a more powerful genetic optimization algorithm: PLANEAT.

    Args:

        x_train (array-like): Training input data.

        y_train (array-like): Labels for training data. one-hot encoded.

        optimizer (function): PLAN optimization technique with hyperparameters. (PLAN using NEAT(PLANEAT) for optimization.) Please use this: from pyerualjetwork import planeat (and) optimizer = lambda *args, **kwargs: planeat.evolve(*args, 'here give your neat hyperparameters for example:  activation_add_prob=0.85', **kwargs) Example:
        ```python
        optimizer = lambda *args, **kwargs: planeat.evolver(*args,
                                                            activation_add_prob=0.05,
                                                            strategy='aggressive',
                                                            policy='more_selective',
                                                            **kwargs)

        model = plan.learner(x_train,
                             y_train,
                             optimizer,
                             fit_start=True,
                             show_history=True,
                             gen=15,
                             batch_size=0.05,
                             interval=16.67)
        ```

        loss (str, optional): For visualizing and monitoring. PLAN neural networks doesn't need any loss function in training. options: ('categorical_crossentropy' or 'binary_crossentropy') Default is 'categorical_crossentropy'.

        target_acc (float, optional): The target accuracy to stop training early when achieved. Default is None.

        target_loss (float, optional): The target loss to stop training early when achieved. Default is None.
        
        fit_start (bool, optional): If the fit_start parameter is set to True, the initial generation population undergoes a simple short training process using the PLAN algorithm. This allows for a very robust starting point, especially for large and complex datasets. However, for small or relatively simple datasets, it may result in unnecessary computational overhead. When fit_start is True, completing the first generation may take slightly longer (this increase in computational cost applies only to the first generation and does not affect subsequent generations). If fit_start is set to False, the initial population will be entirely random. Options: True or False. Default: True

        gen (int, optional): The generation count for genetic optimization.

        batch_size (float, optional): Batch size is used in the prediction process to receive train feedback by dividing the test data into chunks and selecting activations based on randomly chosen partitions. This process reduces computational cost and time while still covering the entire test set due to random selection, so it doesn't significantly impact accuracy. For example, a batch size of 0.08 means each train batch represents 8% of the train set. Default is 1. (%100 of train)

        pop_size (int, optional): Population size of each generation. Default: count of activation functions

        early_stop (bool, optional): If True, implements early stopping during training.(If accuracy not improves in two gen stops learning.) Default is False.

        show_current_activations (bool, optional): Should it display the activations selected according to the current strategies during learning, or not? (True or False) This can be very useful if you want to cancel the learning process and resume from where you left off later. After canceling, you will need to view the live training activations in order to choose the activations to be given to the 'start_this' parameter. Default is False

        show_history (bool, optional): If True, displays the training history after optimization. Default is False.

        auto_normalization (bool, optional): Normalization may solves overflow problem. Default: False

        interval (int, optional): The interval at which evaluations are conducted during training. (33.33 = 30 FPS, 16.67 = 60 FPS) Default is 100.

        target_acc (int, optional): The target accuracy to stop training early when achieved. Default is None.

        start_this_act (list, optional): To resume a previously canceled or interrupted training from where it left off, or to continue from that point with a different strategy, provide the list of activation functions selected up to the learned portion to this parameter. Default is None

        start_this_W (numpy.array, optional): To resume a previously canceled or interrupted training from where it left off, or to continue from that point with a different strategy, provide the weight matrix of this genome. Default is None

        neurons_history (bool, optional): Shows the history of changes that neurons undergo during the TFL (Test or Train Feedback Learning) stages. True or False. Default is False.

        neural_web_history (bool, optional): Draws history of neural web. Default is False.
        
        dtype (numpy.dtype): Data type for the arrays. np.float32 by default. Example: np.float64 or np.float16. [fp32 for balanced devices, fp64 for strong devices, fp16 for weak devices: not reccomended!] (optional)

    Returns:
        tuple: A list for model parameters: [Weight matrix, Test loss, Test Accuracy, [Activations functions]].
    
    """

    from .planeat import define_genomes

    data = 'Train'

    except_this = ['spiral', 'circular']
    activation_potentiation = [item for item in all_activations() if item not in except_this]
    activation_potentiation_len = len(activation_potentiation)
   
    # Pre-checks

    if pop_size is None: pop_size = activation_potentiation_len

    x_train = x_train.astype(dtype, copy=False)
    y_train = optimize_labels(y_train, cuda=False)

    if pop_size < activation_potentiation_len: raise ValueError(f"pop_size must be higher or equal to {activation_potentiation_len}")

    if gen is None: gen = activation_potentiation_len

    if target_acc is not None and (target_acc < 0 or target_acc > 1): raise ValueError('target_acc must be in range 0 and 1')
    if fit_start is not True and fit_start is not False: raise ValueError('fit_start parameter only be True or False. Please read doc-string')

    # Initialize visualization components
    viz_objects = initialize_visualization_for_learner(show_history, neurons_history, neural_web_history, x_train, y_train)

    # Initialize variables
    best_acc = 0
    best_loss = float('inf')
    best_fitness = float('-inf')
    best_acc_per_gen_list = []
    postfix_dict = {}
    loss_list = []
    target_pop = []

    progress = initialize_loading_bar(total=activation_potentiation_len, desc="", ncols=77, bar_format=bar_format_learner)

    if fit_start is False or pop_size > activation_potentiation_len:
        weight_pop, act_pop = define_genomes(input_shape=len(x_train[0]), output_shape=len(y_train[0]), population_size=pop_size, dtype=dtype)

    else:
        weight_pop = [0] * pop_size
        act_pop = [0] * pop_size

    if start_this_act is not None and start_this_W is not None:
        weight_pop[0] = start_this_W
        act_pop[0] = start_this_act

    # LEARNING STARTED
    for i in range(gen):
        postfix_dict["Gen"] = str(i+1) + '/' + str(gen)
        progress.set_postfix(postfix_dict)

        progress.n = 0
        progress.last_print_n = 0
        progress.update(0)

        for j in range(pop_size):

            x_train_batch, y_train_batch = batcher(x_train, y_train, batch_size=batch_size)
            
            if fit_start is True and i == 0 and j < activation_potentiation_len:
                if start_this_act is not None and j == 0:
                    pass
                else:
                    act_pop[j] = activation_potentiation[j]
                    W = fit(x_train_batch, y_train_batch, activation_potentiation=act_pop[j], auto_normalization=auto_normalization, dtype=dtype)
                    weight_pop[j] = W
            
            model = evaluate(x_train_batch, y_train_batch, W=weight_pop[j], activation_potentiation=act_pop[j])
            acc = model[get_acc()]
            
            if loss == 'categorical_crossentropy':
                train_loss = categorical_crossentropy(y_true_batch=y_train_batch, 
                                                           y_pred_batch=model[get_preds_softmax()])
            else:
                train_loss = binary_crossentropy(y_true_batch=y_train_batch, 
                                            y_pred_batch=model[get_preds_softmax()])

            fitness  = wals(acc, train_loss, acc_impact, loss_impact)
            target_pop.append(fitness)

            if fitness >= best_fitness:

                best_fitness = fitness
                best_acc = acc
                best_loss = train_loss
                best_weight = np.copy(weight_pop[j])
                best_model = model

                final_activations = act_pop[j].copy() if isinstance(act_pop[j], list) else act_pop[j]
                final_activations = [final_activations[0]] if len(set(final_activations)) == 1 else final_activations # removing if all same

                if batch_size == 1:
                    postfix_dict[f"{data} Accuracy"] = np.round(best_acc, 4)
                    postfix_dict[f"{data} Loss"] = np.round(train_loss, 4)
                    progress.set_postfix(postfix_dict)

                if show_current_activations:
                    print(f", Current Activations={final_activations}", end='')

                # Update visualizations during training
                if show_history:
                    gen_list = range(1, len(best_acc_per_gen_list) + 2)
                    update_history_plots_for_learner(viz_objects, gen_list, loss_list + [train_loss], 
                                      best_acc_per_gen_list + [best_acc], x_train, final_activations)

                if neurons_history:
                    viz_objects['neurons']['artists'] = (
                        update_neuron_history_for_learner(np.copy(best_weight), viz_objects['neurons']['ax'],
                                     viz_objects['neurons']['row'], viz_objects['neurons']['col'],
                                     y_train[0], viz_objects['neurons']['artists'],
                                     data=data, fig1=viz_objects['neurons']['fig'],
                                     acc=best_acc, loss=train_loss)
                    )

                if neural_web_history:
                    art5_1, art5_2, art5_3 = draw_neural_web(W=best_weight, ax=viz_objects['web']['ax'],
                                                            G=viz_objects['web']['G'], return_objs=True)
                    art5_list = [art5_1] + [art5_2] + list(art5_3.values())
                    viz_objects['web']['artists'].append(art5_list)

                # Check target accuracy
                if target_acc is not None and best_acc >= target_acc:
                    progress.close()
                    train_model = evaluate(x_train, y_train, W=best_weight, 
                                        activation_potentiation=final_activations)
                    if loss == 'categorical_crossentropy':
                        train_loss = categorical_crossentropy(y_true_batch=y_train, 
                                                           y_pred_batch=train_model[get_preds_softmax()])
                    else:
                        train_loss = binary_crossentropy(y_true_batch=y_train, 
                                                       y_pred_batch=train_model[get_preds_softmax()])

                    print('\nActivations: ', final_activations)
                    print(f'Train Accuracy:', train_model[get_acc()])
                    print(f'Train Loss: ', train_loss, '\n')
                    
                    display_visualizations_for_learner(viz_objects, best_weight, data, best_acc, 
                                              best_loss, y_train, interval)
                    return best_weight, best_model[get_preds_softmax()], best_acc, final_activations
            
                # Check target loss
                if target_loss is not None and best_loss <= target_loss:
                    progress.close()
                    train_model = evaluate(x_train, y_train, W=best_weight,
                                        activation_potentiation=final_activations)

                    if loss == 'categorical_crossentropy':
                        train_loss = categorical_crossentropy(y_true_batch=y_train, 
                                                           y_pred_batch=train_model[get_preds_softmax()])
                    else:
                        train_loss = binary_crossentropy(y_true_batch=y_train, 
                                                       y_pred_batch=train_model[get_preds_softmax()])

                    print('\nActivations: ', final_activations)
                    print(f'Train Accuracy :', train_model[get_acc()])
                    print(f'Train Loss : ', train_loss, '\n')

                    # Display final visualizations
                    display_visualizations_for_learner(viz_objects, best_weight, data, best_acc, 
                                              train_loss, y_train, interval)
                    return best_weight, best_model[get_preds_softmax()], best_acc, final_activations

            
            progress.update(1)

        if batch_size != 1:
            train_model = evaluate(x_train, y_train, best_weight, final_activations)
    
            if loss == 'categorical_crossentropy':
                train_loss = categorical_crossentropy(y_true_batch=y_train, 
                                                    y_pred_batch=train_model[get_preds_softmax()])
            else:
                train_loss = binary_crossentropy(y_true_batch=y_train, 
                                                y_pred_batch=train_model[get_preds_softmax()])
            
            postfix_dict[f"{data} Accuracy"] = np.round(train_model[get_acc()], 4)
            postfix_dict[f"{data} Loss"] = np.round(train_loss, 4)
            progress.set_postfix(postfix_dict)
            
            best_acc_per_gen_list.append(train_model[get_acc()])
            loss_list.append(train_loss)

        else:
            best_acc_per_gen_list.append(best_acc)
            loss_list.append(best_loss)

        weight_pop, act_pop = optimizer(np.array(weight_pop, copy=False, dtype=dtype), act_pop, i, np.array(target_pop, dtype=dtype, copy=False), bar_status=False)
        target_pop = []

        # Early stopping check
        if early_stop == True and i > 0:
            if best_acc_per_gen_list[i] == best_acc_per_gen_list[i-1]:
                progress.close()
                train_model = evaluate(x_train, y_train, W=best_weight, 
                                    activation_potentiation=final_activations)
                
            if loss == 'categorical_crossentropy':
                train_loss = categorical_crossentropy(y_true_batch=y_train, 
                                                    y_pred_batch=train_model[get_preds_softmax()])
            else:
                train_loss = binary_crossentropy(y_true_batch=y_train, 
                                                y_pred_batch=train_model[get_preds_softmax()])

            print('\nActivations: ', final_activations)
            print(f'Train Accuracy:', train_model[get_acc()])
            print(f'Train Loss: ', train_loss, '\n')

            # Display final visualizations
            display_visualizations_for_learner(viz_objects, best_weight, data, best_acc, 
                                        train_loss, y_train, interval)
            return best_weight, best_model[get_preds_softmax()], best_acc, final_activations

    # Final evaluation
    progress.close()
    train_model = evaluate(x_train, y_train, W=best_weight,
                        activation_potentiation=final_activations)

    if loss == 'categorical_crossentropy':
        train_loss = categorical_crossentropy(y_true_batch=y_train, 
                                            y_pred_batch=train_model[get_preds_softmax()])
    else:
        train_loss = binary_crossentropy(y_true_batch=y_train, 
                                        y_pred_batch=train_model[get_preds_softmax()])
        
    print('\nActivations: ', final_activations)
    print(f'Train Accuracy:', train_model[get_acc()])
    print(f'Train Loss: ', train_loss, '\n')

    # Display final visualizations
    display_visualizations_for_learner(viz_objects, best_weight, data, best_acc, train_loss, y_train, interval)
    return best_weight, best_model[get_preds_softmax()], best_acc, final_activations


def evaluate(
    x_test,
    y_test,
    W,
    activation_potentiation=['linear']
) -> tuple:
    """
    Evaluates the neural network model using the given test data.

    Args:
        x_test (np.ndarray): Test data.

        y_test (np.ndarray): Test labels (one-hot encoded).

        W (np.ndarray): Neural net weight matrix.
        
        activation_potentiation (list): Activation list. Default = ['linear'].
        
    Returns:
        tuple: Model (list).
    """

    x_test = apply_activation(x_test, activation_potentiation)
    
    result = x_test @ W.T
    
    max_vals = np.max(result, axis=1, keepdims=True)
    
    softmax_preds = np.exp(result - max_vals) / np.sum(np.exp(result - max_vals), axis=1, keepdims=True)
    accuracy = (np.argmax(softmax_preds, axis=1) == np.argmax(y_test, axis=1)).mean()
    
    return W, result, accuracy, None, None, softmax_preds