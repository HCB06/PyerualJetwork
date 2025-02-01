# -*- coding: utf-8 -*-
"""

MAIN MODULE FOR PLAN

Examples: https://github.com/HCB06/PyerualJetwork/tree/main/Welcome_to_PyerualJetwork/ExampleCodes

PLAN document: https://github.com/HCB06/PyerualJetwork/blob/main/Welcome_to_PLAN/PLAN.pdf
PYERUALJETWORK document: https://github.com/HCB06/PyerualJetwork/blob/main/Welcome_to_PyerualJetwork/PYERUALJETWORK_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf

@author: Hasan Can Beydili
@YouTube: https://www.youtube.com/@HasanCanBeydili
@Linkedin: https://www.linkedin.com/in/hasan-can-beydili-77a1b9270/
@Instagram: https://www.instagram.com/canbeydilj/
@contact: tchasancan@gmail.com
"""

import numpy as np
import math

### LIBRARY IMPORTS ###
from .ui import loading_bars, initialize_loading_bar
from .data_operations import normalization, decode_one_hot, batcher
from .loss_functions import binary_crossentropy, categorical_crossentropy
from .activation_functions import apply_activation, Softmax, all_activations
from .metrics import metrics
from .model_operations import get_acc, get_preds, get_preds_softmax
from .memory_operations import optimize_labels
from .visualizations import (
    draw_neural_web,
    update_neural_web_for_fit,
    plot_evaluate,
    update_neuron_history,
    initialize_visualization_for_fit,
    update_weight_visualization_for_fit,
    update_decision_boundary_for_fit,
    update_validation_history_for_fit,
    display_visualization_for_fit,
    display_visualizations_for_learner,
    update_history_plots_for_learner,
    initialize_visualization_for_learner,
    update_neuron_history_for_learner,
    show
)

### GLOBAL VARIABLES ###
bar_format_normal = loading_bars()[0]
bar_format_learner = loading_bars()[1]

# BUILD -----

def fit(
    x_train,
    y_train,
    val=False,
    val_count=None,
    activation_potentiation=['linear'],
    x_val=None,
    y_val=None,
    show_training=None,
    interval=100,
    LTD=0,
    decision_boundary_status=True,
    train_bar=True,
    auto_normalization=True,
    neurons_history=False,
    dtype=np.float32
):
    """
    Creates a model to fitting data.
    
    fit Args:

        x_train (list[num]): List or numarray of input data.

        y_train (list[num]): List or numarray of target labels. (one hot encoded)

        val (None or True): validation in training process ? None or True default: None (optional)

        val_count (None or int): After how many examples learned will an accuracy test be performed? default: 10=(%10) it means every approximately 10 step (optional)

        activation_potentiation (list): For deeper PLAN networks, activation function parameters. For more information please run this code: plan.activations_list() default: [None] (optional)

        x_val (list[num]): List of validation data. default: x_train (optional)

        y_val (list[num]): (list[num]): List of target labels. (one hot encoded) default: y_train (optional)

        show_training (bool, str): True or None default: None (optional)

        LTD (int): Long Term Depression Hyperparameter for train PLAN neural network default: 0 (optional)

        interval (float, int): frame delay (milisecond) parameter for Training Report (show_training=True) This parameter effects to your Training Report performance. Lower value is more diffucult for Low end PC's (33.33 = 30 FPS, 16.67 = 60 FPS) default: 100 (optional)

        decision_boundary_status (bool): If the visualization of validation and training history is enabled during training, should the decision boundaries also be visualized? True or False. Default is True. (optional)

        train_bar (bool): Training loading bar? True or False. Default is True. (optional)

        auto_normalization(bool): Normalization process during training. May effect training time and model quality. True or False. Default is True. (optional)

        neurons_history (bool, optional): Shows the history of changes that neurons undergo during the CL (Cumulative Learning) stages. True or False. Default is False. (optional)

        dtype (numpy.dtype): Data type for the arrays. np.float32 by default. Example: np.float64 or np.float16. [fp32 for balanced devices, fp64 for strong devices, fp16 for weak devices: not reccomended!] (optional)

    Returns:
        numpyarray([num]): (Weight matrix).
    """

    # Pre-checks

    x_train = x_train.astype(dtype, copy=False)

    if train_bar and val:
        train_progress = initialize_loading_bar(total=len(x_train), ncols=71, desc='Fitting', bar_format=bar_format_normal)
    elif train_bar and val == False:
        train_progress = initialize_loading_bar(total=len(x_train), ncols=44, desc='Fitting', bar_format=bar_format_normal)

    if len(x_train) != len(y_train):
        raise ValueError("x_train and y_train must have the same length.")

    if val and (x_val is None and y_val is None):
        x_val, y_val = x_train, y_train

    elif val and (x_val is not None and y_val is not None):
        x_val = x_val.astype(dtype, copy=False)
        y_val = y_val.astype(dtype, copy=False)

    val_list = [] if val else None
    val_count = val_count or 10
    # Defining weights
    STPW = np.ones((len(y_train[0]), len(x_train[0].ravel()))).astype(dtype, copy=False)  # STPW = SHORT TIME POTENTIATION WEIGHT
    LTPW = np.zeros((len(y_train[0]), len(x_train[0].ravel()))).astype(dtype, copy=False)  # LTPW = LONG TIME POTENTIATION WEIGHT
    # Initialize visualization
    vis_objects = initialize_visualization_for_fit(val, show_training, neurons_history, x_train, y_train)

    # Training process
    for index, inp in enumerate(x_train):
        inp = np.array(inp, copy=False).ravel()
        y_decoded = decode_one_hot(y_train[index])
        # Weight updates
        STPW = feed_forward(inp, STPW, is_training=True, Class=y_decoded, activation_potentiation=activation_potentiation, LTD=LTD)
        LTPW += normalization(STPW, dtype=dtype) if auto_normalization else STPW
        if val and index != 0:
            if index % math.ceil((val_count / len(x_train)) * 100) == 0:
                val_acc = evaluate(x_val, y_val, loading_bar_status=False, activation_potentiation=activation_potentiation, W=LTPW)[get_acc()]
                val_list.append(val_acc)

                # Visualization updates
                if show_training:
                    update_weight_visualization_for_fit(vis_objects['ax'][0, 0], LTPW, vis_objects['artist2'])
                    if decision_boundary_status:
                        update_decision_boundary_for_fit(vis_objects['ax'][0, 1], x_val, y_val, activation_potentiation, LTPW, vis_objects['artist1'])
                    update_validation_history_for_fit(vis_objects['ax'][1, 1], val_list, vis_objects['artist3'])
                    update_neural_web_for_fit(W=LTPW, G=vis_objects['G'], ax=vis_objects['ax'][1, 0], artist=vis_objects['artist4'])
                if neurons_history:
                    update_neuron_history(LTPW, row=vis_objects['row'], col=vis_objects['col'], class_count=len(y_train[0]), fig1=vis_objects['fig1'], ax1=vis_objects['ax1'], artist5=vis_objects['artist5'], acc=val_acc)
        if train_bar:
             train_progress.update(1)

        STPW = np.ones((len(y_train[0]), len(x_train[0].ravel()))).astype(dtype, copy=False)

    # Finalize visualization
    if show_training:
        ani1 = display_visualization_for_fit(vis_objects['fig'], vis_objects['artist1'], interval)
        ani2 = display_visualization_for_fit(vis_objects['fig'], vis_objects['artist2'], interval)
        ani3 = display_visualization_for_fit(vis_objects['fig'], vis_objects['artist3'], interval)
        ani4 = display_visualization_for_fit(vis_objects['fig'], vis_objects['artist4'], interval)
        show()
    
    if neurons_history:
        ani5 = display_visualization_for_fit(vis_objects['fig1'], vis_objects['artist5'], interval)
        show()

    return normalization(LTPW, dtype=dtype)


def learner(x_train, y_train, optimizer, fit_start, strategy='accuracy', gen=None, batch_size=1,
           neural_web_history=False, show_current_activations=False, auto_normalization=True,
           neurons_history=False, early_stop=False, loss='categorical_crossentropy', show_history=False,
           interval=33.33, target_acc=None, target_loss=None,
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
        genetic_optimizer = lambda *args, **kwargs: planeat.evolver(*args,
                                                                    activation_add_prob=0.85,
                                                                    strategy='aggressive',
                                                                    **kwargs)

        model = plan.learner(x_train,
                             y_train,
                             optimizer=genetic_optimizer,
                             fit_start=True,
                             strategy='accuracy',
                             show_history=True,
                             gen=15,
                             batch_size=0.05,
                             interval=16.67)
        ```

        fit_start (bool): If the fit_start parameter is set to True, the initial generation population undergoes a simple short training process using the PLAN algorithm. This allows for a very robust starting point, especially for large and complex datasets. However, for small or relatively simple datasets, it may result in unnecessary computational overhead. When fit_start is True, completing the first generation may take slightly longer (this increase in computational cost applies only to the first generation and does not affect subsequent generations). If fit_start is set to False, the initial population will be entirely random. Options: True or False. The fit_start parameter is MANDATORY and must be provided.

        strategy (str, optional): Learning strategy. (options: 'accuracy', 'f1', 'precision', 'recall'): 'accuracy', Maximizes train (or test if given) accuracy during learning. 'f1', Maximizes train (or test if given) f1 score during learning. 'precision', Maximizes train (or test if given) precision score during learning. 'recall', Maximizes train (or test if given) recall during learning. Default is 'accuracy'.

        gen (int, optional): The generation count for genetic optimization.

        batch_size (float, optional): Batch size is used in the prediction process to receive train feedback by dividing the test data into chunks and selecting activations based on randomly chosen partitions. This process reduces computational cost and time while still covering the entire test set due to random selection, so it doesn't significantly impact accuracy. For example, a batch size of 0.08 means each train batch represents 8% of the train set. Default is 1. (%100 of train)

        early_stop (bool, optional): If True, implements early stopping during training.(If accuracy not improves in two gen stops learning.) Default is False.

        auto_normalization (bool, optional): IMPORTANT: auto_nomralization parameter works only if fit_start is True. Do not change this value if fit_start is False, because it doesnt matter.) If auto normalization=False this makes more faster training times and much better accuracy performance for some datasets. Default is True.

        show_current_activations (bool, optional): Should it display the activations selected according to the current strategies during learning, or not? (True or False) This can be very useful if you want to cancel the learning process and resume from where you left off later. After canceling, you will need to view the live training activations in order to choose the activations to be given to the 'start_this' parameter. Default is False

        show_history (bool, optional): If True, displays the training history after optimization. Default is False.

        loss (str, optional): For visualizing and monitoring. PLAN neural networks doesn't need any loss function in training. options: ('categorical_crossentropy' or 'binary_crossentropy') Default is 'categorical_crossentropy'.

        interval (int, optional): The interval at which evaluations are conducted during training. (33.33 = 30 FPS, 16.67 = 60 FPS) Default is 100.

        target_acc (int, optional): The target accuracy to stop training early when achieved. Default is None.

        target_loss (float, optional): The target loss to stop training early when achieved. Default is None.
        
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

    activation_potentiation = all_activations()
    activation_potentiation_len = len(activation_potentiation)
   
    # Pre-checks

    x_train = x_train.astype(dtype, copy=False)
    y_train = optimize_labels(y_train, cuda=False)

    if gen is None:
        gen = activation_potentiation_len

    if strategy != 'accuracy' and strategy != 'f1' and strategy != 'recall' and strategy != 'precision': raise ValueError("Strategy parameter only be 'accuracy' or 'f1' or 'recall' or 'precision'.")
    if target_acc is not None and (target_acc < 0 or target_acc > 1): raise ValueError('target_acc must be in range 0 and 1')
    if fit_start is not True and fit_start is not False: raise ValueError('fit_start parameter only be True or False. Please read doc-string')

    # Initialize visualization components
    viz_objects = initialize_visualization_for_learner(show_history, neurons_history, neural_web_history, x_train, y_train)

    # Initialize progress bar
    if batch_size == 1:
        ncols = 76
    else:
        ncols = 89

    # Initialize variables
    best_acc = 0
    best_f1 = 0
    best_recall = 0
    best_precision = 0 
    best_acc_per_gen_list = []
    postfix_dict = {}
    loss_list = []
    target_pop = []

    progress = initialize_loading_bar(total=activation_potentiation_len, desc="", ncols=ncols, bar_format=bar_format_learner)

    if fit_start is False:
        weight_pop, act_pop = define_genomes(input_shape=len(x_train[0]), output_shape=len(y_train[0]), population_size=activation_potentiation_len, dtype=dtype)
    
        if start_this_act is not None and start_this_W is not None:
            weight_pop[0] = start_this_W
            act_pop[0] = start_this_act

    else:
        weight_pop = []
        act_pop = []

    for i in range(gen):
        postfix_dict["Gen"] = str(i+1) + '/' + str(gen)
        progress.set_postfix(postfix_dict)

        progress.n = 0
        progress.last_print_n = 0
        progress.update(0)

        for j in range(activation_potentiation_len):

            x_train_batch, y_train_batch = batcher(x_train, y_train, batch_size=batch_size)
            
            if fit_start is True and i == 0:
                act_pop.append(activation_potentiation[j])
                W = fit(x_train_batch, y_train_batch, activation_potentiation=act_pop[-1], train_bar=False, auto_normalization=auto_normalization, dtype=dtype)
                weight_pop.append(W)
            
            model = evaluate(x_train_batch, y_train_batch, W=weight_pop[j], loading_bar_status=False, activation_potentiation=act_pop[j], dtype=dtype)
            acc = model[get_acc()]
            
            if strategy == 'accuracy': target_pop.append(acc)

            elif strategy == 'f1' or strategy == 'precision' or strategy == 'recall':
                precision_score, recall_score, f1_score = metrics(y_train_batch, model[get_preds()])

                if strategy == 'precision':
                    target_pop.append(precision_score)

                    if i == 0 and j == 0:
                        best_precision = precision_score

                if strategy == 'recall':
                    target_pop.append(recall_score)

                    if i == 0 and j == 0:
                        best_recall = recall_score

                if strategy == 'f1':
                    target_pop.append(f1_score)

                    if i == 0 and j == 0:
                        best_f1 = f1_score

            if ((strategy == 'accuracy' and acc >= best_acc) or 
                (strategy == 'f1' and f1_score >= best_f1) or 
                (strategy == 'precision' and precision_score >= best_precision) or
                (strategy == 'recall' and recall_score >= best_recall)):

                best_acc = acc
                best_weights = np.copy(weight_pop[j])
                final_activations = act_pop[j].copy() if isinstance(act_pop[j], list) else act_pop[j]
                best_model = model

                final_activations = [final_activations[0]] if len(set(final_activations)) == 1 else final_activations # removing if all same

                if batch_size == 1:
                    postfix_dict[f"{data} Accuracy"] = best_acc
                else:
                    postfix_dict[f"{data} Batch Accuracy"] = acc
                progress.set_postfix(postfix_dict)

                if show_current_activations:
                    print(f", Current Activations={final_activations}", end='')

                if loss == 'categorical_crossentropy':
                    train_loss = categorical_crossentropy(y_true_batch=y_train_batch, y_pred_batch=model[get_preds_softmax()])
                else:
                    train_loss = binary_crossentropy(y_true_batch=y_train_batch, y_pred_batch=model[get_preds_softmax()])

                if batch_size == 1:
                    postfix_dict[f"{data} Loss"] = train_loss
                    best_loss = train_loss
                else:
                    postfix_dict[f"{data} Batch Loss"] = train_loss
                progress.set_postfix(postfix_dict)
                best_loss = train_loss

                # Update visualizations during training
                if show_history:
                    gen_list = range(1, len(best_acc_per_gen_list) + 2)
                    update_history_plots_for_learner(viz_objects, gen_list, loss_list + [train_loss], 
                                      best_acc_per_gen_list + [best_acc], x_train, final_activations)

                if neurons_history:
                    viz_objects['neurons']['artists'] = (
                        update_neuron_history_for_learner(np.copy(best_weights), viz_objects['neurons']['ax'],
                                     viz_objects['neurons']['row'], viz_objects['neurons']['col'],
                                     y_train[0], viz_objects['neurons']['artists'],
                                     data=data, fig1=viz_objects['neurons']['fig'],
                                     acc=best_acc, loss=train_loss)
                    )

                if neural_web_history:
                    art5_1, art5_2, art5_3 = draw_neural_web(W=best_weights, ax=viz_objects['web']['ax'],
                                                            G=viz_objects['web']['G'], return_objs=True)
                    art5_list = [art5_1] + [art5_2] + list(art5_3.values())
                    viz_objects['web']['artists'].append(art5_list)

                # Check target accuracy
                if target_acc is not None and best_acc >= target_acc:
                    progress.close()
                    train_model = evaluate(x_train, y_train, W=best_weights, loading_bar_status=False, 
                                        activation_potentiation=final_activations, dtype=dtype)

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
                    display_visualizations_for_learner(viz_objects, best_weights, data, best_acc, 
                                              train_loss, y_train, interval)
                    return best_weights, best_model[get_preds()], best_acc, final_activations

                # Check target loss
                if target_loss is not None and best_loss <= target_loss:
                    progress.close()
                    train_model = evaluate(x_train, y_train, W=best_weights, loading_bar_status=False, 
                                        activation_potentiation=final_activations, dtype=dtype)

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
                    display_visualizations_for_learner(viz_objects, best_weights, data, best_acc, 
                                              train_loss, y_train, interval)
                    return best_weights, best_model[get_preds()], best_acc, final_activations

            progress.update(1)

        best_acc_per_gen_list.append(best_acc)
        loss_list.append(best_loss)

        weight_pop, act_pop = optimizer(np.array(weight_pop, copy=False, dtype=dtype), act_pop, i, np.array(target_pop, dtype=dtype, copy=False), bar_status=False)
        target_pop = []

        # Early stopping check
        if early_stop == True and i > 0:
            if best_acc_per_gen_list[i] == best_acc_per_gen_list[i-1]:
                progress.close()
                train_model = evaluate(x_train, y_train, W=best_weights, loading_bar_status=False, 
                                    activation_potentiation=final_activations, dtype=dtype)

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
                display_visualizations_for_learner(viz_objects, best_weights, data, best_acc, 
                                          train_loss, y_train, interval)
                return best_weights, best_model[get_preds()], best_acc, final_activations

    # Final evaluation
    progress.close()
    train_model = evaluate(x_train, y_train, W=best_weights, loading_bar_status=False, 
                        activation_potentiation=final_activations, dtype=dtype)

    if loss == 'categorical_crossentropy':
        train_loss = categorical_crossentropy(y_true_batch=y_train, y_pred_batch=train_model[get_preds_softmax()])
    else:
        train_loss = binary_crossentropy(y_true_batch=y_train, y_pred_batch=train_model[get_preds_softmax()])

    print('\nActivations: ', final_activations)
    print(f'Train Accuracy :', train_model[get_acc()])
    print(f'Train Loss : ', train_loss, '\n')

    # Display final visualizations
    display_visualizations_for_learner(viz_objects, best_weights, data, best_acc, train_loss, y_train, interval)
    return best_weights, best_model[get_preds()], best_acc, final_activations



def feed_forward(
    Input,               # list[num]: Input data.
    w,                   # num: Weight matrix of the neural network.
    is_training,        # bool: Flag indicating if the function is called during training (True or False).
    activation_potentiation,
    Class='?',               # int: Which class is, if training. # (list): Activation potentiation list for deep PLAN. (optional)
    LTD=0
) -> tuple:
    """
    Applies feature extraction process to the input data using synaptic potentiation.

    Args:
        Input (num): Input data.
        w (num): Weight matrix of the neural network.
        is_training (bool): Flag indicating if the function is called during training (True or False).
        Class (int): if is during training then which class(label) ? is isnt then put None.
        # activation_potentiation (list): ac list for deep PLAN. default: [None] ('linear') (optional)

    Returns:
        tuple: A tuple (vector) containing the neural layer result and the updated weight matrix.
        or
        num: neural network output
    """
    
    Output = apply_activation(Input, activation_potentiation)

    Input = Output

    if is_training == True:
        
        for _ in range(LTD):        

            depression_vector = np.random.rand(*Input.shape)

            Input -= depression_vector

        w[Class, :] = Input
        return w

    else:

        neural_layer = np.dot(w, Input)

        return neural_layer


def evaluate(
    x_test,         # NumPy array: Test input data.
    y_test,         # NumPy array: Test labels.
    W,              # List of NumPy arrays: Neural network weight matrices.
    activation_potentiation=['linear'], # List of activation functions.
    loading_bar_status=True,  # Optionally show loading bar.
    show_metrics=None,        # Optionally show metrics.
    dtype=np.float32
) -> tuple:
    """
    Evaluates the neural network model using the given test data.

    Args:
        x_test (np.ndarray): Test input data.
        
        y_test (np.ndarray): Test labels. one-hot encoded.
        
        W (list[np.ndarray]): List of neural network weight matrices.
        
        activation_potentiation (list): List of activation functions.
        
        loading_bar_status (bool): Option to show a loading bar (optional).
        
        show_metrics (bool): Option to show metrics (optional).

        dtype (numpy.dtype): Data type for the arrays. np.float32 by default. Example: np.float64 or np.float16. [fp32 for balanced devices, fp64 for strong devices, fp16 for weak devices: not reccomended!]

    Returns:
        tuple: Predicted labels, model accuracy, and other evaluation metrics.
    """
    # Pre-checks

    x_test = x_test.astype(dtype, copy=False)

    if len(y_test[0]) < 256:
        if y_test.dtype != np.uint8:
            y_test = np.array(y_test, copy=False).astype(np.uint8, copy=False)
    elif len(y_test[0]) <= 32767:
        if y_test.dtype != np.uint16:
            y_test = np.array(y_test, copy=False).astype(np.uint16, copy=False)
    else:
        if y_test.dtype != np.uint32:
            y_test = np.array(y_test, copy=False).astype(np.uint32, copy=False)

    predict_probabilitys = np.empty((len(x_test), W.shape[0]), dtype=dtype)
    real_classes = np.empty(len(x_test), dtype=y_test.dtype)     
    predict_classes = np.empty(len(x_test), dtype=y_test.dtype)   

    true_predict = 0
    acc_list = np.empty(len(x_test), dtype=dtype)    

    if loading_bar_status:
        loading_bar = initialize_loading_bar(total=len(x_test), ncols=64, desc='Testing', bar_format=bar_format_normal)

    for inpIndex in range(len(x_test)):
        Input = x_test[inpIndex].ravel() 

        neural_layer = Input

        neural_layer = feed_forward(neural_layer, np.copy(W), is_training=False, Class='?', activation_potentiation=activation_potentiation)

        predict_probabilitys[inpIndex] = Softmax(neural_layer)

        RealOutput = np.argmax(y_test[inpIndex])
        real_classes[inpIndex] = RealOutput
        PredictedOutput = np.argmax(neural_layer)
        predict_classes[inpIndex] = PredictedOutput

        if RealOutput == PredictedOutput:
            true_predict += 1

        acc = true_predict / (inpIndex + 1)  
        acc_list[inpIndex] = acc

        if loading_bar_status:
            loading_bar.update(1)
            loading_bar.set_postfix({"Test Accuracy": acc})

    if loading_bar_status:
        loading_bar.close()

    if show_metrics:
        # Plot the evaluation metrics
        plot_evaluate(x_test, y_test, predict_classes, acc_list, W=np.copy(W), activation_potentiation=activation_potentiation)

    return W, predict_classes, acc_list[-1], None, None, predict_probabilitys