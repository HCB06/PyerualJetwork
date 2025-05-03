# -*- coding: utf-8 -*-
""" 


NN (Neural Networks)
============================
This module hosts functions for training and evaluating artificial neural networks for labeled classification tasks.

Currently, 3 types of models can be trained:

    PLAN (Potentiation Learning Artificial Neural Network)
        * Training Time for Small Projects: fast
        * Training Time for Big Projects: fast
        * Explainability: high
        * Learning Capacity: medium (compared to single perceptrons)

    MLP (Multi-Layer Perceptron → Deep Learning) -- With non-bias
        * Training Time for Small Projects: fast
        * Training Time for Big Projects: slow
        * Explainability: low
        * Learning Capacity: high

    PTNN (Potentiation Transfer Neural Network) -- With non-bias
        * Training Time for Small Projects: fast
        * Training Time for Big Projects: fast
        * Explainability: low
        * Learning Capacity: high

Read learn function docstring for know how to use of these model architectures.

For more information about PLAN: https://github.com/HCB06/PyerualJetwork/blob/main/Welcome_to_PLAN/PLAN.pdf

Module functions:
-----------------
- plan_fit()
- learn()
- evaluate()

Examples: https://github.com/HCB06/PyerualJetwork/tree/main/Welcome_to_PyerualJetwork/ExampleCodes

PyerualJetwork document: https://github.com/HCB06/PyerualJetwork/blob/main/Welcome_to_PyerualJetwork/PYERUALJETWORK_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf

- Creator: Hasan Can Beydili
- YouTube: https://www.youtube.com/@HasanCanBeydili
- Linkedin: https://www.linkedin.com/in/hasan-can-beydili-77a1b9270/
- Instagram: https://www.instagram.com/canbeydilj
- Contact: tchasancan@gmail.com
"""

import numpy as np
import cupy as cp
import copy
import random

### LIBRARY IMPORTS ###
from .ui import loading_bars, initialize_loading_bar
from .cpu.activation_functions import all_activations
from .model_ops import get_acc, get_preds_softmax, get_preds
from .memory_ops import optimize_labels, transfer_to_gpu
from .fitness_functions import wals

### GLOBAL VARIABLES ###
bar_format_normal = loading_bars()[0]
bar_format_learner = loading_bars()[1]

# BUILD -----

def plan_fit(
    x_train,
    y_train,
    activations=['linear'],
    W=None,
    auto_normalization=False,
    cuda=False,
    dtype=np.float32
):
    """
    Creates a PLAN model to fitting data.
    
    Args:
        x_train (aray-like[num or cup]): List or numarray of input data.

        y_train (aray-like[num or cup]): List or numarray of target labels. (one hot encoded)

        activations (list): For deeper PLAN networks, activation function parameters. For more information please run this code: neu.activations_list() default: [None] (optional)

        W (numpy.ndarray or cupy.ndarray): If you want to re-continue or update model

        auto_normalization (bool, optional): Normalization may solves overflow problem. Default: False

        cuda (bool, optional): CUDA GPU acceleration ? Default = False.
        
        dtype (numpy.dtype): Data type for the arrays. np.float32 by default. Example: np.float64 or np.float16.

    Returns:
        numpyarray: (Weight matrix).
    """
    
    from .cpu.data_ops import normalization
    if not cuda:
        from cpu.activation_functions import apply_activation
        array_type = np

    else:
        from .cuda.activation_functions import apply_activation
        array_type = cp
        
        
    # Pre-check

    if len(x_train) != len(y_train): raise ValueError("x_train and y_train must have the same length.")

    weight = array_type.zeros((len(y_train[0]), len(x_train[0].ravel()))).astype(dtype, copy=False) if W is None else W

    if auto_normalization is True: x_train = normalization(apply_activation(x_train, activations))
    elif auto_normalization is False: x_train = apply_activation(x_train, activations)
    else: raise ValueError('normalization parameter only be True or False')

    weight += y_train.T @ x_train if not cuda else cp.array(y_train).T @ cp.array(x_train)

    return normalization(weight.get() if cuda else weight,dtype=dtype)


def learn(x_train, y_train, optimizer, gen, pop_size, fit_start=True, batch_size=1,
           weight_evolve=True, neural_web_history=False, show_current_activations=False, auto_normalization=False,
           neurons_history=False, early_stop=False, show_history=False, target_loss=None,
           interval=33.33, target_acc=None, loss='categorical_crossentropy', acc_impact=0.9, loss_impact=0.1,
           start_this_act=None, start_this_W=None, neurons=[], activation_functions=[], decision_boundary_history=False, cuda=False, dtype=np.float32):
    """
    Optimizes the activation functions for a neural network by leveraging train data to find 
    the most accurate combination of activation potentiation(or activation function) & weight values for the labeled classificaiton dataset.

    Why genetic optimization ENE(Eugenic NeuroEvolution) and not backpropagation?
    Because PLAN is different from other neural network architectures. In PLAN, the learnable parameters are not the weights; instead, the learnable parameters are the activation functions.
    Since activation functions are not differentiable, we cannot use gradient descent or backpropagation. However, I developed a more powerful genetic optimization algorithm: ENE.

    * This function also able to train classic MLP model architectures.
    * And my newest innovative architecture: PTNN (Potentiation Transfer Neural Network).

    Examples:
    
        This creates a PLAN model: 
            - ```model = learn(x_train, y_train, optimizer, template_model, pop_size=100, gen=100, fit_start=True) ```

        This creates a MLP model(with 2 hidden layer): 
            - ```model = learn(x_train, y_train, optimizer, template_model, pop_size=100, gen=100, fit_start=False, neurons=[64, 64], activation_functions=['tanh', 'tanh']) ```
        
        This creates a PTNN model(with 2 hidden layer & 1 aggregation layer(comes with PLAN)): 
           -  ```model = learn(x_train, y_train, optimizer, template_model, pop_size=100, gen=[10, 100], fit_start=True, neurons=[64, 64], activation_functions=['tanh', 'tanh']) ```
    
    :Args:
    :param x_train: (array-like): Training input data.
    :param y_train: (array-like): Labels for training data. one-hot encoded.
    :param optimizer: (function): Optimization technique with hyperparameters. (PLAN, MLP & PTNN (all) using ENE for optimization. Gradient based technique's will added in the future.) Please use this: from pyerualjetwork.ene import evolver (and) optimizer = lambda *args, **kwargs: evolver(*args, 'here give your hyperparameters for example:  activation_add_prob=0.85', **kwargs) Example:
    ```python
             optimizer = lambda *args, **kwargs: ene.evolver(*args,
                                                            activation_add_prob=0.05,
                                                            strategy='aggressive',
                                                            policy='more_selective',
                                                            **kwargs)

             model = nn.learn(x_train,
                            y_train,
                            optimizer,
                            fit_start=True,
                            show_history=True,
                            gen=15,
                            batch_size=0.05,
                            interval=16.67)
    ```
    :param fit_start: (bool, optional): If the fit_start parameter is set to True, the initial generation population undergoes a simple short training process using the PLAN algorithm. This allows for a very robust starting point, especially for large and complex datasets. However, for small or relatively simple datasets, it may result in unnecessary computational overhead. When fit_start is True, completing the first generation may take slightly longer (this increase in computational cost applies only to the first generation and does not affect subsequent generations). If fit_start is set to False, the initial population will be entirely random. Additonaly if you want to train PTNN model you must be give True. Options: True or False. Default: True
    :param gen: (int or list): The generation count for genetic optimization. If you want to train PTNN model you must give a list of two number. First number for PLAN model training second number for MLP.
    :param batch_size: (float, optional): Batch size is used in the prediction process to receive train feedback by dividing the train data into chunks and selecting activations based on randomly chosen partitions. This process reduces computational cost and time while still covering the entire train set due to random selection, so it doesn't significantly impact accuracy. For example, a batch size of 0.08 means each train batch represents %8 of the train set. Default is 1. (%100 of train)
    :param pop_size: (int): Population size of each generation.
    :param weight_evolve: (bool, optional): Activation combinations already optimizes by ENE genetic search algorithm. Should the weight parameters also evolve or should the weights be determined according to the aggregating learning principle of the PLAN algorithm? Default: True (Evolves Weights)
    :param neural_web_history: (bool, optional): Draws history of neural web. Default is False. [ONLY FOR PLAN MODELS]
    :param show_current_activations: (bool, optional): Should it display the activations selected according to the current strategies during learning, or not? (True or False) This can be very useful if you want to cancel the learning process and resume from where you left off later. After canceling, you will need to view the live training activations in order to choose the activations to be given to the 'start_this' parameter. Default is False
    :param auto_normalization: (bool, optional): Normalization may solves overflow problem. Default: False
    :param neurons_history: (bool, optional): Shows the history of changes that neurons undergo during the ENE optimization process. True or False. Default is False. [ONLY FOR PLAN MODELS]
    :param early_stop: (bool, optional): If True, implements early stopping during training.(If accuracy not improves in two gen stops learning.) Default is False.
    :param show_history: (bool, optional): If True, displays the training history after optimization. Default is False.
    :param target_loss: (float, optional): The target loss to stop training early when achieved. Default is None.
    :param interval: (int, optional): The interval at which evaluations are conducted during training. (33.33 = 30 FPS, 16.67 = 60 FPS) Default is 100.
    :param target_acc: (float, optional): The target accuracy to stop training early when achieved. Default is None.
    :param loss: (str, optional): options: ('categorical_crossentropy' or 'binary_crossentropy') Default is 'categorical_crossentropy'.
    :param acc_impact: (float, optional): Impact of accuracy for optimization [0-1]. Default: 0.9
    :param loss_impact: (float, optional): Impact of loss for optimization [0-1]. Default: 0.1
    :param start_this_act: (list, optional): To resume a previously canceled or interrupted training from where it left off, or to continue from that point with a different strategy, provide the list of activation functions selected up to the learned portion to this parameter. Default is None
    :param start_this_W: (numpy.array, optional): To resume a previously canceled or interrupted training from where it left off, or to continue from that point with a different strategy, provide the weight matrix of this genome. Default is None
    :param neurons: (list[int], optional): If you dont want train PLAN model this parameter represents neuron count of each hidden layer for MLP or PTNN. Number of elements --> Layer count. Default: [] (No hidden layer) --> architecture setted to PLAN, if not --> architecture setted to MLP.
    :param decision_boundary_history: (bool, optional): At the end of the training, the decision boundary history is shown in animation form. Note: If you are sure of your memory, set it to True. Default: False
    :param activation_functions: (list[str], optional): If you dont want train PLAN model this parameter represents activation function of each hidden layer for MLP or PTNN. if neurons is not [] --> uses default: ['linear'] * len(neurons). if neurons is [] --> uses [].
    :param dtype: (numpy.dtype): Data type for the Weight matrices. np.float32 by default. Example: np.float64 or np.float16.
    :param cuda: (bool, optional): CUDA GPU acceleration ? Default = False.

    Returns:
        tuple: A list for model parameters: [Weight matrix, Train Preds, Train Accuracy, [Activations functions]].
    """

    from .ene import define_genomes
    from .cpu.visualizations import display_decision_boundary_history, create_decision_boundary_hist, plot_decision_boundary
    from .model_ops import get_model_template

    if cuda is False:
        from .cpu.data_ops import batcher 
        from .cpu.loss_functions import categorical_crossentropy, binary_crossentropy
        from .cpu.visualizations import (
        draw_neural_web,
        display_visualizations_for_learner,
        update_history_plots_for_learner,
        initialize_visualization_for_learner,
        update_neuron_history_for_learner)
    
    else:
        from .cuda.data_ops import batcher
        from .cuda.loss_functions import categorical_crossentropy, binary_crossentropy
        from .cuda.visualizations import (
        draw_neural_web,
        display_visualizations_for_learner,
        update_history_plots_for_learner,
        initialize_visualization_for_learner,
        update_neuron_history_for_learner)
        
    data = 'Train'

    template_model = get_model_template()

    except_this = ['spiral', 'circular']
    activations = [item for item in all_activations() if item not in except_this]
    activations_len = len(activations)

    def format_number(val):
        if abs(val) >= 1e4 or (abs(val) < 1e-2 and val != 0):
            return f"{val:.4e}"
        else:
            return f"{val:.4f}"

    # Pre-checks

    if cuda: 
        x_train = transfer_to_gpu(x_train, dtype=dtype)
        y_train = transfer_to_gpu(y_train, dtype=dtype)
        
    if pop_size > activations_len and fit_start is True:
        for _ in range(pop_size - len(activations)):
            random_index_all_act = random.randint(0, len(activations)-1)
            activations.append(activations[random_index_all_act])

    y_train = optimize_labels(y_train, cuda=cuda)

    if pop_size < activations_len: raise ValueError(f"pop_size must be higher or equal to {activations_len}")

    if target_acc is not None and (target_acc < 0 or target_acc > 1): raise ValueError('target_acc must be in range 0 and 1')
    if fit_start is not True and fit_start is not False: raise ValueError('fit_start parameter only be True or False. Please read doc-string')

    if neurons != []:
        weight_evolve = True
        
        if activation_functions == []: activation_functions = ['linear'] * len(neurons)
        
        if fit_start is False:
            # MLP
            activations = activation_functions
            model_type = 'MLP'
            activation_potentiations = [0] * pop_size
            activation_potentiation = None
            is_mlp = True
            transfer_learning = False
        else:
            # PTNN
            model_type = 'PLAN' # First generation index gen[0] is PLAN, other index gen[1] it will change to PTNN (PLAN Connects to MLP and will transfer the learned information).
            transfer_learning = True

            neurons_copy = neurons.copy()
            neurons = []
            gen_copy = gen.copy()
            gen = gen[0] + gen[1]
            activation_potentiations = [0] * pop_size
            activation_potentiation = None
            is_mlp = False # it will change
        
    else:
        # PLAN
        model_type = 'PLAN'
        transfer_learning = False

        activation_potentiations = [0] * pop_size # NOTE: For PLAN models, activation_potentiations is needed BUT activations variable already mirros activation_potentiation values.
                                      # So, we don't need to use activation_potentiations variable. activation_potentiations variable is only for PTNN models.
        activation_potentiation = None
        is_mlp = False


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

    progress = initialize_loading_bar(total=pop_size, desc="", ncols=85, bar_format=bar_format_learner)

    if fit_start is False:
        weight_pop, act_pop = define_genomes(input_shape=len(x_train[0]), output_shape=len(y_train[0]), neurons=neurons, activation_functions=activations, population_size=pop_size, dtype=dtype)

    else:
        weight_pop = [0] * len(activations)
        act_pop = [0] * len(activations)

    if start_this_act is not None and start_this_W is not None:
        weight_pop[0] = start_this_W
        act_pop[0] = start_this_act

    # LEARNING STARTED
    for i in range(gen):

      # TRANSFORMATION PLAN TO MLP FOR PTNN (in later generations)
        if model_type == 'PLAN' and transfer_learning:
            if i == gen_copy[0]:
                
                model_type = 'PTNN'
                neurons = neurons_copy

                for individual in range(len(weight_pop)):
                    weight_pop[individual] = np.copy(best_weight)
                    activation_potentiations[individual] = final_activations.copy() if isinstance(final_activations, list) else final_activations

                activation_potentiation = activation_potentiations[0]

                neurons_copy = [len(y_train[0])] + neurons_copy
                activation_functions = ['linear'] + activation_functions
                
                weight_pop, act_pop = define_genomes(input_shape=len(x_train[0]), output_shape=len(y_train[0]), neurons=neurons_copy, activation_functions=activation_functions, population_size=pop_size, dtype=dtype)
 
                # 0 indexed individual will keep PLAN's learned informations and in later generations it will share other individuals.
                for l in range(1, len(weight_pop[0])):
                    original_shape = weight_pop[0][l].shape

                    identity_matrix = np.eye(original_shape[0], original_shape[1], dtype=weight_pop[0][l].dtype)
                    weight_pop[0][l] = identity_matrix
                    
                for l in range(len(weight_pop)):
                    weight_pop[l][0] = np.copy(best_weight)

                best_weight = np.array(weight_pop[0], dtype=object)
                final_activations = act_pop[0]
                is_mlp = True
                fit_start = False


        postfix_dict["Gen"] = str(i+1) + '/' + str(gen)
        progress.set_postfix(postfix_dict)

        progress.n = 0
        progress.last_print_n = 0
        progress.update(0)

        for j in range(pop_size):
            
            x_train_batch, y_train_batch = batcher(x_train, y_train, batch_size=batch_size)

            if fit_start is True and i == 0:
                if start_this_act is not None and j == 0:
                    pass
                else:
           
                    act_pop[j] = activations[j]
                    W = plan_fit(x_train_batch, y_train_batch, activations=act_pop[j], cuda=cuda, auto_normalization=auto_normalization, dtype=dtype)
                    weight_pop[j] = W


            if weight_evolve is False:
                weight_pop[j] = plan_fit(x_train_batch, y_train_batch, activations=act_pop[j], cuda=cuda, auto_normalization=auto_normalization, dtype=dtype)


            model = evaluate(x_train_batch, y_train_batch, W=weight_pop[j], activations=act_pop[j], activation_potentiations=activation_potentiations[j], auto_normalization=auto_normalization, cuda=cuda, model_type=model_type)
            acc = model[get_acc()]
            
            if loss == 'categorical_crossentropy':
                train_loss = categorical_crossentropy(y_true_batch=y_train_batch, 
                                                           y_pred_batch=model[get_preds_softmax()])
            else:
                train_loss = binary_crossentropy(y_true_batch=y_train_batch, 
                                            y_pred_batch=model[get_preds_softmax()])

            fitness = wals(acc, train_loss, acc_impact, loss_impact)
            target_pop.append(fitness if not cuda else fitness.get())
            
            if fitness >= best_fitness:

                best_fitness = fitness
                best_acc = acc
                best_loss = train_loss
                best_weight = np.copy(weight_pop[j]) if model_type == 'PLAN' else copy.deepcopy(weight_pop[j])
                best_model = model

                if isinstance(act_pop[j], list) and model_type == 'PLAN':
                    final_activations = act_pop[j].copy()
                elif isinstance(act_pop[j], str):
                    final_activations = act_pop[j]
                else:
                    final_activations = copy.deepcopy(act_pop[j])
                    
                if model_type == 'PLAN': final_activations = [final_activations[0]] if len(set(final_activations)) == 1 else final_activations # removing if all same

                if batch_size == 1:
                    postfix_dict[f"{data} Accuracy"] = format_number(best_acc)
                    postfix_dict[f"{data} Loss"] = format_number(train_loss)
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
                                        activations=final_activations, activation_potentiations=activation_potentiation, auto_normalization=auto_normalization, cuda=cuda, model_type=model_type)
                    if loss == 'categorical_crossentropy':
                        train_loss = categorical_crossentropy(y_true_batch=y_train, 
                                                           y_pred_batch=train_model[get_preds_softmax()])
                    else:
                        train_loss = binary_crossentropy(y_true_batch=y_train, 
                                                       y_pred_batch=train_model[get_preds_softmax()])

                    print('\nActivations: ', final_activations)
                    print('Activation Potentiation: ', activation_potentiation)
                    print('Train Accuracy:', train_model[get_acc()])
                    print('Train Loss: ', train_loss, '\n')
                    print('Model Type:', model_type)
                    
                    if decision_boundary_history: display_decision_boundary_history(fig, artist, interval=interval)

                    display_visualizations_for_learner(viz_objects, best_weight, data, best_acc, 
                                              best_loss, y_train, interval)
                    
                    model = template_model._replace(weights=best_weight,
                                           predictions=best_model[get_preds()],
                                           accuracy=best_acc,
                                           activations=final_activations,
                                           softmax_predictions=best_model[get_preds_softmax()],
                                           model_type=model_type,
                                           activation_potentiation=activation_potentiation)
                    
                    return model
            
                # Check target loss
                if target_loss is not None and best_loss <= target_loss:
                    progress.close()
                    train_model = evaluate(x_train, y_train, W=best_weight,
                                        activations=final_activations, activation_potentiations=activation_potentiation, auto_normalization=auto_normalization, cuda=cuda, model_type=model_type)

                    if loss == 'categorical_crossentropy':
                        train_loss = categorical_crossentropy(y_true_batch=y_train, 
                                                           y_pred_batch=train_model[get_preds_softmax()])
                    else:
                        train_loss = binary_crossentropy(y_true_batch=y_train, 
                                                       y_pred_batch=train_model[get_preds_softmax()])

                    print('\nActivations: ', final_activations)
                    print('Activation Potentiation: ', activation_potentiation)
                    print('Train Accuracy:', train_model[get_acc()])
                    print('Train Loss: ', train_loss, '\n')
                    print('Model Type:', model_type)

                    if decision_boundary_history: display_decision_boundary_history(fig, artist, interval=interval)

                    # Display final visualizations
                    display_visualizations_for_learner(viz_objects, best_weight, data, best_acc, 
                                              train_loss, y_train, interval)
                    
                    model = template_model._replace(weights=best_weight,
                                           predictions=best_model[get_preds()],
                                           accuracy=best_acc,
                                           activations=final_activations,
                                           softmax_predictions=best_model[get_preds_softmax()],
                                           model_type=model_type,
                                           activation_potentiation=activation_potentiation)
                    
                    return model

            
            progress.update(1)

        if decision_boundary_history:
            if i == 0: 
                fig, ax = create_decision_boundary_hist()
                artist = []
            artist = plot_decision_boundary(x_train_batch, y_train_batch, activations=act_pop[np.argmax(target_pop)], W=weight_pop[np.argmax(target_pop)], cuda=cuda, model_type=model_type, ax=ax, artist=artist)

        if batch_size != 1:
            train_model = evaluate(x_train, y_train, W=best_weight, activations=final_activations, activation_potentiations=activation_potentiation, auto_normalization=auto_normalization, cuda=cuda, model_type=model_type)
    
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

        if model_type == 'PLAN': weight_pop = np.array(weight_pop, copy=False, dtype=dtype)
        else: weight_pop = np.array(weight_pop, copy=False, dtype=object)

        weight_pop, act_pop = optimizer(weight_pop, act_pop, i, np.array(target_pop, dtype=dtype, copy=False), weight_evolve=weight_evolve, is_mlp=is_mlp, bar_status=False)
        target_pop = []

        # Early stopping check
        if early_stop == True and i > 0:
            if best_acc_per_gen_list[i] == best_acc_per_gen_list[i-1]:
                progress.close()
                train_model = evaluate(x_train, y_train, W=best_weight, 
                                    activations=final_activations, activation_potentiations=activation_potentiation, auto_normalization=auto_normalization, cuda=cuda, model_type=model_type)
                
            if loss == 'categorical_crossentropy':
                train_loss = categorical_crossentropy(y_true_batch=y_train, 
                                                    y_pred_batch=train_model[get_preds_softmax()])
            else:
                train_loss = binary_crossentropy(y_true_batch=y_train, 
                                                y_pred_batch=train_model[get_preds_softmax()])

            print('\nActivations: ', final_activations)
            print('Activation Potentiation: ', activation_potentiation)
            print('Train Accuracy:', train_model[get_acc()])
            print('Train Loss: ', train_loss, '\n')
            print('Model Type:', model_type)

            if decision_boundary_history: display_decision_boundary_history(fig, artist, interval=interval)

            # Display final visualizations
            display_visualizations_for_learner(viz_objects, best_weight, data, best_acc, 
                                        train_loss, y_train, interval)
            
            model = template_model._replace(weights=best_weight,
                                    predictions=best_model[get_preds()],
                                    accuracy=best_acc,
                                    activations=final_activations,
                                    softmax_predictions=best_model[get_preds_softmax()],
                                    model_type=model_type,
                                    activation_potentiation=activation_potentiation)

            return model

    # Final evaluation
    progress.close()

    train_model = evaluate(x_train, y_train, W=best_weight,
                        activations=final_activations, activation_potentiations=activation_potentiation, auto_normalization=auto_normalization, cuda=cuda, model_type=model_type)

    if loss == 'categorical_crossentropy':
        train_loss = categorical_crossentropy(y_true_batch=y_train, 
                                            y_pred_batch=train_model[get_preds_softmax()])
    else:
        train_loss = binary_crossentropy(y_true_batch=y_train, 
                                        y_pred_batch=train_model[get_preds_softmax()])
        

    print('\nActivations: ', final_activations)
    print('Activation Potentiation: ', activation_potentiation)
    print('Train Accuracy:', train_model[get_acc()])
    print('Train Loss: ', train_loss, '\n')
    print('Model Type:', model_type)

    if decision_boundary_history: display_decision_boundary_history(fig, artist, interval=interval)

    # Display final visualizations
    display_visualizations_for_learner(viz_objects, best_weight, data, best_acc, train_loss, y_train, interval)
    
    model = template_model._replace(weights=best_weight,
                            predictions=best_model[get_preds()],
                            accuracy=best_acc,
                            activations=final_activations,
                            softmax_predictions=best_model[get_preds_softmax()],
                            model_type=model_type,
                            activation_potentiation=activation_potentiation)

    return model


def evaluate(
    x_test,
    y_test,
    model=None,
    model_type=None,
    W=None,
    activations=['linear'],
    activation_potentiations=[],
    auto_normalization=False,
    show_report=False,
    cuda=False
) -> tuple:
    """
    Evaluates the neural network model using the given test data.

    Args:
        x_test (np.ndarray): Test data.

        y_test (np.ndarray): Test labels (one-hot encoded).

        model (tuple, optional): Trained model.

        model_type: (str, optional): Type of the model. Options: 'PLAN', 'MLP', 'PTNN'.

        W (array-like, optional): Neural net weight matrix.

        activations (list, optional): Activation list for PLAN or MLP models (MLP layers activations if it PTNN model). Default = ['linear'].

        activation_potentiations (list, optional): Extra activation potentiation list (PLAN layers activations) for PTNN models. Default = [].

        auto_normalization (bool, optional): Normalization for x_test ? Default = False.

        show_report (bool, optional): show test report. Default = False

        cuda (bool, optional): CUDA GPU acceleration ? Default = False.

    Returns:
        tuple: W, preds, accuracy, None, None, softmax_preds
    """

    from .cpu.visualizations import plot_evaluate
    from .model_ops import predict_from_memory

    if not cuda:
        from .cpu.data_ops import normalization
        
        if model:
            sample_acc = model.accuracy
            if hasattr(sample_acc, "get"): model = model._replace(accuracy=sample_acc.get())

    else:
        from .cuda.data_ops import normalization
        x_test = cp.array(x_test)
        y_test = cp.array(y_test)

        if model:
            sample_acc = model.accuracy
            if isinstance(sample_acc, np.number): model = model._replace(accuracy=cp.array(sample_acc))

    if model is None:
        from .model_ops import get_model_template
        template_model = get_model_template()
        
        model = template_model._replace(weights=W,
                        activations=activations,
                        model_type=model_type,
                        activation_potentiation=activation_potentiations)
    
    result = predict_from_memory(x_test, model, cuda=cuda)

    if auto_normalization: x_test = normalization(x_test, dtype=x_test.dtype)

    array_type = cp if cuda else np

    max_vals = array_type.max(result, axis=1, keepdims=True)
    
    softmax_preds = array_type.exp(result - max_vals) / array_type.sum(array_type.exp(result - max_vals), axis=1, keepdims=True)
    accuracy = (array_type.argmax(softmax_preds, axis=1) == array_type.argmax(y_test, axis=1)).mean()

    if show_report: plot_evaluate(x_test, y_test, result, acc=accuracy, model=model, cuda=cuda)

    return W, result, accuracy, None, None, softmax_preds