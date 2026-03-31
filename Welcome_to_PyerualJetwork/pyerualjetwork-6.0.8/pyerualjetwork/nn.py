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
        * Explainability: medium
        * Learning Capacity: high

Read learn function docstring for know how to use of these model architectures.

For more information about PLAN: https://github.com/HCB06/PyerualJetwork/blob/main/Welcome_to_PLAN/PLAN.pdf

Module functions:
-----------------
- plan_fit()
- learn()
- grad()
- evaluate()

Examples: https://github.com/HCB06/PyerualJetwork/tree/main/Welcome_to_PyerualJetwork/ExampleCodes

PyerualJetwork document: https://github.com/HCB06/PyerualJetwork/blob/main/Welcome_to_PyerualJetwork/PYERUALJETWORK_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf

- Creator: Hasan Can Beydili
- YouTube: https://www.youtube.com/@HasanCanBeydili
- Linkedin: https://www.linkedin.com/in/hasan-can-beydili-77a1b9270/
- Instagram: https://www.instagram.com/canbeydili
- Contact: tchasancan@gmail.com
"""

import numpy as np
import copy
import random
from multiprocessing import Pool, cpu_count

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
        from .cpu.activation_functions import apply_activation
        array_type = np

    else:
        from .cuda.activation_functions import apply_activation
        import cupy as cp
        array_type = cp
        
        
    # Pre-check

    if len(x_train) != len(y_train): raise ValueError("x_train and y_train must have the same length.")

    weight = array_type.zeros((len(y_train[0]), len(x_train[0].ravel()))).astype(dtype, copy=False) if W is None else W

    if auto_normalization is True: x_train = normalization(apply_activation(x_train, activations))
    elif auto_normalization is False: x_train = apply_activation(x_train, activations)
    else: raise ValueError('normalization parameter only be True or False')

    weight += y_train.T @ x_train if not cuda else array_type.array(y_train).T @ array_type.array(x_train)

    return normalization(weight.get() if cuda else weight,dtype=dtype)

def learn(x_train, y_train, iter, genetic_optimizer=None, gradient_optimizer=None, pop_size=None, fit_start=True, batch_size=1,
           weight_evolve=True, neural_web_history=False, show_current_activations=False, auto_normalization=False,
           neurons_history=False, early_stop=False, show_history=False, target_loss=None,
           interval=33.33, target_acc=None, loss='categorical_crossentropy', acc_impact=0.9, loss_impact=0.1,
           start_this_act=None, start_this_W=None, neurons=[], activation_functions=[], parallel_training=False,
           thread_count=cpu_count(), decision_boundary_history=False, cuda=False, quick_start=False,
           backprop_train=False,
           step_size=32,
           dtype=np.float32):
    """
    Optimizes the activation functions (and optionally weights) for a neural network by leveraging
    train data to find the most accurate combination of activation potentiation (or activation function)
    & weight values for the labeled classification dataset.

    Why genetic optimization FBN (Fitness Biased NeuroEvolution) and not backpropagation?
    Because PLAN is different from other neural network architectures. In PLAN, the learnable parameters
    are not the weights; instead, the learnable parameters are the activation functions.
    Since activation functions are not differentiable, we cannot use gradient descent or backpropagation.
    However, I developed a more powerful genetic optimization algorithm: ENE.

    * This function also able to train classic MLP model architectures.
    * And my newest innovative architecture: PTNN (Potentiation Transfer Neural Network).
    * Gradient-based training is now supported via the backprop_train flag and gradient_optimizer.

    ── OPTIMIZER PARAMETERS — IMPORTANT CHANGE ───────────────────────────────
    The old single 'optimizer' parameter has been split into two distinct parameters:
      • genetic_optimizer  — ENE-based (evolutionary) training  (PLAN / MLP / PTNN)
      • gradient_optimizer — gradient-based (backprop) training
    You must supply the correct optimizer for the training mode you choose.
    ──────────────────────────────────────────────────────────────────────────

    Examples:

        This creates a PLAN model:
        ```python
        genetic_optimizer = lambda *args, **kwargs: ene.evolver(*args,
                                                                activation_add_prob=0.05,
                                                                strategy='aggressive',
                                                                policy='more_selective',
                                                                **kwargs)

        model = nn.learn(x_train, y_train, genetic_optimizer=genetic_optimizer,
                         iter=100, pop_size=100, fit_start=True)
        ```

        This creates a MLP model with 2 hidden layers (genetic):
        ```python
        genetic_optimizer = lambda *args, **kwargs: ene.evolver(*args,
                                                                activation_add_prob=0.05,
                                                                strategy='aggressive',
                                                                policy='more_selective',
                                                                **kwargs)

        model = nn.learn(x_train, y_train, genetic_optimizer=genetic_optimizer,
                         iter=100, pop_size=100, fit_start=False, batch_size=0.2,
                         neurons=[64, 64], activation_functions=['tanh', 'tanh'])
        ```

        This creates a MLP model trained with 2 hidden layers (backprop):
        ```python
        gradient_optimizer = lambda *args, **kwargs: nn.grad(*args,
                                                            method='adam',
                                                            learning_rate=0.001,
                                                            momentum=0.9',
                                                            beta1=0.9,
                                                            **kwargs)

        model = nn.learn(x_train, y_train, gradient_optimizer=gradient_optimizer,
                         iter=50, backprop_train=True,
                         neurons=[64, 64], activation_functions=['relu', 'relu'],
                         step_size=32)
        ```


        This creates a PTNN 2 hidden layers + 1 PLAN aggregation layer model with fine-tuning after the PLAN phase (genetic):
        ```python
        genetic_optimizer = lambda *args, **kwargs: ene.evolver(*args,
                                                        activation_add_prob=0.05,
                                                        strategy='aggressive',
                                                        policy='more_selective',
                                                        **kwargs)

        model = nn.learn(x_train, y_train, genetic_optimizer=genetic_optimizer,
                         iter=[10, 100], pop_size=100, fit_start=True, batch_size=0.2,
                         neurons=[64, 64], activation_functions=['tanh', 'tanh'])
        ```

        This creates a PTNN 2 hidden layers + 1 PLAN aggregation layer model with fine-tuning after the PLAN phase (backprop):
        ```python
        gradient_optimizer = lambda *args, **kwargs: nn.grad(*args,
                                                    method='adam',
                                                    learning_rate=0.001,
                                                    momentum=0.9',
                                                    beta1=0.9,
                                                    **kwargs)

        genetic_optimizer = lambda *args, **kwargs: ene.evolver(*args,
                                                    activation_add_prob=0.05,
                                                    strategy='aggressive',
                                                    policy='more_selective',
                                                    **kwargs)

        model = nn.learn(x_train, y_train, genetic_optimizer=genetic_optimizer,
                         gradient_optimizer=adam_optimizer,
                         iter=[10, 50], pop_size=100, fit_start=True,
                         backprop_train=True,
                         neurons=[64, 64], activation_functions=['relu', 'relu'],
                         step_size=64)
        ```

    ── MIGRATION NOTES (breaking changes from previous version) ──────────────
    • 'gen'       → renamed to 'iter'
    • 'optimizer' → split into 'genetic_optimizer' and 'gradient_optimizer'

    Old: learn(x_train, y_train, optimizer=opt, gen=100, pop_size=50)
    New: learn(x_train, y_train, genetic_optimizer=opt, iter=100, pop_size=50)
    ──────────────────────────────────────────────────────────────────────────

    :Args:
    :param x_train: (array-like): Training input data.
    :param y_train: (array-like): Labels for training data. one-hot encoded.
    :param iter: (int or list): The iteration/generation count for optimization. For PLAN and MLP models provide a single integer (e.g. iter=100). For PTNN models provide a list of two integers [plan_iters, mlp_iters] — first value controls the PLAN phase, second controls the MLP phase (genetic or gradient). Previously named 'gen' — update your code accordingly.
    :param genetic_optimizer: (callable, optional): ENE-based evolutionary optimizer with hyperparameters pre-bound. Required when backprop_train=False. Use a lambda to bind hyperparameters. Example:
    ```python
        genetic_optimizer = lambda *args, **kwargs: ene.evolver(*args,
                                                                activation_add_prob=0.05,
                                                                strategy='aggressive',
                                                                policy='more_selective',
                                                                **kwargs)

        model = nn.learn(x_train, y_train, genetic_optimizer=genetic_optimizer,
                         iter=15, pop_size=100, batch_size=0.05,
                         fit_start=True, show_history=True, interval=16.67)
    ```
    :param gradient_optimizer: (callable, optional): Gradient-based optimizer (e.g. Adam, SGD) used when backprop_train=True. Must accept the signature: (y_train, softmax_preds, cache, weights, activations, state) -> (new_weights, new_state). Required when backprop_train=True. Default is None. Example:
    ```python
        gradient_optimizer = lambda *args, **kwargs: nn.grad(*args,
                                                    method='adam',
                                                    learning_rate=0.001,
                                                    momentum=0.9',
                                                    beta1=0.9,
                                                    **kwargs)

        model = nn.learn(x_train, y_train, gradient_optimizer=gradient_optimizer,
                         iter=15, pop_size=100, batch_size=0.05,
                         fit_start=True, show_history=True, interval=16.67)
    ```
    :param pop_size: (int, optional): Population size of each generation for genetic optimization. Not required when backprop_train=True on a pure MLP. Default is None (sum of activation functions count).
    :param fit_start: (bool, optional): If the fit_start parameter is set to True, the initial generation population undergoes a simple short training process using the PLAN algorithm. This allows for a very robust starting point, especially for large and complex datasets. However, for small or relatively simple datasets, it may result in unnecessary computational overhead. When fit_start is True, completing the first generation may take slightly longer (this increase in computational cost applies only to the first generation and does not affect subsequent generations). If fit_start is set to False, the initial population will be entirely random. Additionally if you want to train a PTNN model you must set this to True. Options: True or False. Default: True
    :param batch_size: (float, optional): Batch size is used in the prediction process to receive train feedback by dividing the train data into chunks and selecting activations based on randomly chosen partitions. This process reduces computational cost and time while still covering the entire train set due to random selection, so it doesn't significantly impact accuracy. For example, a batch size of 0.08 means each train batch represents %8 of the train set. Default is 1 (%100 of train). Note: Not used in gradient training — use step_size instead.
    :param weight_evolve: (bool, optional): Activation combinations are already optimized by the ENE genetic search algorithm. Should the weight parameters also evolve or should the weights be determined according to the aggregating learning principle of the PLAN algorithm? Default: True (Evolves Weights)
    :param neural_web_history: (bool, optional): Draws history of neural web. Default is False. [ONLY FOR PLAN MODELS]
    :param show_current_activations: (bool, optional): Should it display the activations selected according to the current strategies during learning, or not? (True or False) This can be very useful if you want to cancel the learning process and resume from where you left off later. After canceling, you will need to view the live training activations in order to choose the activations to be given to the 'start_this_act' parameter. Default is False
    :param auto_normalization: (bool, optional): Normalization may solve overflow problems. Default: False
    :param neurons_history: (bool, optional): Shows the history of changes that neurons undergo during the ENE optimization process. True or False. Default is False. [ONLY FOR PLAN MODELS]
    :param early_stop: (bool, optional): If True, implements early stopping during training. (If accuracy does not improve for two consecutive iterations, stops learning.) Default is False.
    :param show_history: (bool, optional): If True, displays the training history after optimization. Default is False.
    :param target_loss: (float, optional): The target loss to stop training early when achieved. Default is None.
    :param interval: (float, optional): The refresh interval in milliseconds for live visualizations. (33.33 = 30 FPS, 16.67 = 60 FPS) Default is 33.33.
    :param target_acc: (float, optional): The target accuracy (0-1) to stop training early when achieved. Default is None.
    :param loss: (str, optional): Loss function to use. Options: 'categorical_crossentropy' or 'binary_crossentropy'. Default is 'categorical_crossentropy'.
    :param acc_impact: (float, optional): Impact of accuracy on the WALS fitness score [0-1]. Default: 0.9
    :param loss_impact: (float, optional): Impact of loss on the WALS fitness score [0-1]. Default: 0.1
    :param start_this_act: (list, optional): To resume a previously canceled or interrupted training from where it left off, or to continue from that point with a different strategy, provide the list of activation functions selected up to the learned portion to this parameter. Default is None
    :param start_this_W: (numpy.ndarray, optional): To resume a previously canceled or interrupted training from where it left off, or to continue from that point with a different strategy, provide the weight matrix of this genome. Default is None
    :param neurons: (list[int], optional): Neuron count of each hidden layer for MLP or PTNN models. Number of elements = layer count. Default: [] (no hidden layers) → architecture set to PLAN. If non-empty → architecture set to MLP (or PTNN when fit_start=True).
    :param activation_functions: (list[str], optional): Activation function of each hidden layer for MLP or PTNN. If neurons is non-empty and this is left empty, defaults to ['linear'] * len(neurons). If neurons is [] this is ignored.
    :param parallel_training: (bool, optional): If True, evaluates population members in parallel using multiprocessing. Recommended for large pop_size values to significantly speed up training. Default = False. For example code: https://github.com/HCB06/PyerualJetwork/blob/main/Welcome_to_PyerualJetwork/ExampleCodes/iris_multi_thread(mlp).py
    :param thread_count: (int, optional): Maximum number of parallel worker processes when parallel_training=True. Default: automatically selects the maximum thread count of your system.
    :param decision_boundary_history: (bool, optional): At the end of training, the decision boundary history is shown in animation form. Note: Memory-intensive for large datasets. Default: False
    :param cuda: (bool, optional): CUDA GPU acceleration. Default = False. Recommended for large neural network architectures or big datasets to significantly speed up training.
    :param quick_start: (bool, optional): If True, accelerates the PLAN phase of a PTNN model by copying the best individual's weights across the entire population after first generation. Converges faster at the cost of population diversity. Default: False
    :param backprop_train: (bool, optional): If True, switches from genetic optimization to gradient-based (backpropagation) training. For pure MLP trains with mini-batch gradient descent from the start. For PTNN the PLAN phase still uses ENE and the MLP phase switches to gradient descent. Requires gradient_optimizer to be set. Uses step_size (not batch_size) to control mini-batch size. Default: False
    :param step_size: (int, optional): Mini-batch size for gradient training. Must be a positive integer (e.g. 32, 64, 128) — analogous to batch_size in TensorFlow/Keras. step_size=1 is treated as full-batch gradient descent. Ignored when backprop_train=False. Default: 32
    :param dtype: (numpy.dtype): Data type for the weight matrices. Default: np.float32. Other options: np.float64, np.float16.

    Returns:
        NamedTuple: A model object with fields:
            - weights               : Optimized weight matrices.
            - predictions           : Train set predictions (argmax).
            - accuracy              : Final train accuracy (float, 0-1).
            - activations           : Selected/final activation functions.
            - softmax_predictions   : Raw softmax output probabilities.
            - model_type            : 'PLAN', 'MLP', or 'PTNN'.
            - activation_potentiation: Activation potentiation values (PLAN/PTNN only).
    """
    from .ene import define_genomes
    from .cpu.visualizations import display_decision_boundary_history, create_decision_boundary_hist, plot_decision_boundary
    from .model_ops import get_model_template

    if cuda is False:
        from .cpu.data_ops import batcher
        from .cpu.loss_functions import categorical_crossentropy, binary_crossentropy
        from .cpu.visualizations import (
            draw_neural_web, display_visualizations_for_learner,
            update_history_plots_for_learner, initialize_visualization_for_learner,
            update_neuron_history_for_learner)
    else:
        from .cuda.data_ops import batcher
        from .cuda.loss_functions import categorical_crossentropy, binary_crossentropy
        from .cuda.visualizations import (
            draw_neural_web, display_visualizations_for_learner,
            update_history_plots_for_learner, initialize_visualization_for_learner,
            update_neuron_history_for_learner)

    data = 'Train'
    template_model = get_model_template()

    except_this = ['spiral', 'circular']
    # The above code is a comment in Python. Comments are used to provide explanations or notes within
    # the code and are not executed by the interpreter. In this case, the comment appears to be
    # indicating that the code below it is related to "activations".
    activations = [item for item in all_activations() if item not in except_this]
    activations_len = len(activations)

    def format_number(val):
        if abs(val) >= 1e4 or (abs(val) < 1e-3 and val != 0):
            return f"{val:.4e}"
        else:
            return f"{val:.4f}"
    
  # ── gradient train için ortak yardımcı ───────────────────────────────────
    def _run_backprop_train(init_weights, grad_epochs, final_activations, activation_potentiation):
        """
        Mini-batch gradient descent döngüsü.
        step_size sadece integer kabul eder (32, 64, 128 gibi - TensorFlow'daki batch_size gibi).
        
        Loading bar: total=grad_epochs — bar her EPOCH'ta bir adım ilerler.
        Batch adımlarında sadece postfix güncellenir (bar kıpırdamaz).
        Epoch bitince Train Accuracy/Loss yazılır, bar bir adım ilerler.
        """
        
        grad_weights = [np.copy(w) for w in init_weights]
        grad_state   = None
        global_step  = 0

        # ── Batch boyutunu belirle (step_size parametresinden) ─────────────────
        N_total = len(x_train)
        
        # step_size kontrolü - sadece integer kabul et
        if not isinstance(step_size, int):
            raise TypeError(f"step_size integer olmalı, {type(step_size)} verildi")
        
        if step_size <= 0:
            raise ValueError(f"step_size pozitif olmalı, {step_size} verildi")
        
        bs = min(step_size, N_total)
        
        # steps_per_epoch'u doğru hesapla (ceil bölme - tüm veriyi kapsamalı)
        steps_per_epoch = (N_total + bs - 1) // bs
        if step_size == 1: 
            is_full_batch = True
        else:
            is_full_batch = False
        
        print(f"Step size: {bs}, Steps per epoch: {steps_per_epoch}, Full-batch: {is_full_batch}")

        # ── Bar: total=grad_epochs, epoch başına 1 adım ──────────────────────
        progress = initialize_loading_bar(
            total=grad_epochs,
            desc="",
            ncols=85,
            bar_format=bar_format_learner
        )
        postfix_dict = {}

        train_model = train_loss = train_acc = best_weight = None

        for epoch in range(1, grad_epochs + 1):

            postfix_dict["Epoch"] = f"{epoch}/{grad_epochs}"
            progress.set_postfix(postfix_dict)

            # ── Epoch başında karıştır ────────────────────────────────────────
            if not is_full_batch:
                perm   = np.random.permutation(N_total)
                x_shuf = x_train[perm]
                y_shuf = y_train[perm]
            else:
                x_shuf = x_train
                y_shuf = y_train

            for step in range(steps_per_epoch):

                global_step += 1

                # ── Mini-batch ────────────────────────────────────────────────
                start   = step * bs
                end     = min(start + bs, N_total)
                x_batch = x_shuf[start:end]
                y_batch = y_shuf[start:end]

                # ── define_genomes formatına çevir ────────────────────────────
                best_weight = np.empty(len(grad_weights), dtype=object)
                for idx, w in enumerate(grad_weights):
                    best_weight[idx] = w

                # ── İleri geçiş + cache ───────────────────────────────────────
                batch_model = evaluate(
                    x_batch, y_batch,
                    W=best_weight,
                    activations=final_activations,
                    activation_potentiations=activation_potentiation,
                    auto_normalization=auto_normalization,
                    cuda=cuda,
                    model_type=model_type,
                    return_cache=True,
                )
                softmax_preds = batch_model[get_preds_softmax()]
                cache         = batch_model[6]

                if loss == 'categorical_crossentropy':
                    batch_loss = categorical_crossentropy(
                        y_true_batch=y_batch, y_pred_batch=softmax_preds)
                else:
                    batch_loss = binary_crossentropy(
                        y_true_batch=y_batch, y_pred_batch=softmax_preds)


                # ── Backprop + optimizer ──────────────────────────────────────
                grad_weights, grad_state = gradient_optimizer(
                    y_train=np.array(y_batch, dtype=np.float32),
                    softmax_preds=np.array(softmax_preds, dtype=np.float32),
                    cache=cache,
                    weights=grad_weights,
                    activations=final_activations,
                    state=grad_state,
                    cuda=cuda
                )

            # ── EPOCH SONU ────────────────────────────────────────────────────
            # full-batch: son batch == tam set, ekstra evaluate yok
            # mini-batch: tam set üzerinde evaluate
            if not is_full_batch:
                best_weight = np.empty(len(grad_weights), dtype=object)
                for idx, w in enumerate(grad_weights):
                    best_weight[idx] = w

                train_model = evaluate(
                    x_train, y_train,
                    W=best_weight,
                    activations=final_activations,
                    activation_potentiations=activation_potentiation,
                    auto_normalization=auto_normalization,
                    cuda=cuda,
                    model_type=model_type,
                )
                if loss == 'categorical_crossentropy':
                    train_loss = categorical_crossentropy(
                        y_true_batch=y_train,
                        y_pred_batch=train_model[get_preds_softmax()])
                else:
                    train_loss = binary_crossentropy(
                        y_true_batch=y_train,
                        y_pred_batch=train_model[get_preds_softmax()])
                train_acc = float(train_model[get_acc()])
            else:
                train_model = batch_model
                train_loss  = batch_loss
                train_acc   = float(batch_model[get_acc()])

            # Batch etiketlerini kaldır, Train etiketlerini yaz, bar bir ileri
            postfix_dict["Train Accuracy"] = format_number(train_acc)
            postfix_dict["Train Loss"]     = format_number(float(train_loss))
            progress.set_postfix(postfix_dict)
            progress.update(1)   # ← bar sadece burada ilerler (epoch başına 1 adım)

            # ── Erken durdurma ────────────────────────────────────────────────
            if target_acc is not None and train_acc >= target_acc:
                progress.close()
                return template_model._replace(
                    weights=best_weight,
                    predictions=train_model[get_preds()],
                    accuracy=train_acc,
                    activations=final_activations,
                    softmax_predictions=train_model[get_preds_softmax()],
                    model_type=model_type,
                    activation_potentiation=activation_potentiation)

            if target_loss is not None and train_loss <= target_loss:
                progress.close()
                return template_model._replace(
                    weights=best_weight,
                    predictions=train_model[get_preds()],
                    accuracy=train_acc,
                    activations=final_activations,
                    softmax_predictions=train_model[get_preds_softmax()],
                    model_type=model_type,
                    activation_potentiation=activation_potentiation)

        progress.close()

        print('\nActivations: ', final_activations)
        print('Activation Potentiation: ', activation_potentiation)
        print('Train Accuracy:', train_model[get_acc()])
        print('Train Loss: ', train_loss, '\n')
        print('Model Type:', model_type)

        return template_model._replace(
            weights=best_weight,
            predictions=train_model[get_preds()],
            accuracy=train_acc,
            activations=final_activations,
            softmax_predictions=train_model[get_preds_softmax()],
            model_type=model_type,
            activation_potentiation=activation_potentiation)

    # ─────────────────────────────────────────────────────────────────────────

    # Pre-checks
    if cuda:
        x_train = transfer_to_gpu(x_train, dtype=dtype)
        y_train = transfer_to_gpu(y_train, dtype=dtype)
    
    if pop_size is None and len(activations) % 2 == 0: 
        pop_size = len(activations)
    elif pop_size is None and len(activations) % 2 != 0:
        pop_size = len(activations) + 1

    if pop_size > activations_len and fit_start is True:
        for _ in range(pop_size - len(activations)):
            activations.append(activations[random.randint(0, len(activations) - 1)])

    y_train = optimize_labels(y_train, cuda=cuda)

    if not backprop_train:
        if pop_size < activations_len:
            raise ValueError(f"pop_size must be higher or equal to {activations_len}")

    if target_acc is not None and (target_acc < 0 or target_acc > 1):
        raise ValueError('target_acc must be in range 0 and 1')
    if fit_start is not True and fit_start is not False:
        raise ValueError('fit_start parameter only be True or False. Please read doc-string')

    if neurons != []:
        weight_evolve = True
        if activation_functions == []:
            activation_functions = ['linear'] * len(neurons)

        if fit_start is False:
            activations = activation_functions
            model_type = 'MLP'
            activation_potentiations = [0] * pop_size
            activation_potentiation = None
            is_mlp = True
            transfer_learning = False
        else:
            model_type = 'PLAN'
            transfer_learning = True
            neurons_copy = neurons.copy()
            neurons = []
            iter_copy = iter.copy()
            iter = iter[0] + iter[1]
            activation_potentiations = [0] * pop_size
            activation_potentiation = None
            is_mlp = False

            if neurons_copy[0] < len(y_train[0]):
                raise ValueError(
                    f"For PTNN architecture, neurons[0] (fist layer neuron count): ({neurons_copy[0]}) must be "
                    f"greater than or equal to output_size ({len(y_train[0])}). "
                    f"Please set neurons[0] >= {len(y_train[0])}."
                )
    
    else:
        model_type = 'PLAN'
        transfer_learning = False
        activation_potentiations = [0] * pop_size
        activation_potentiation = None
        is_mlp = False

    # ── Saf MLP + backprop_train için xavier başlatma ──────────────────────────────────────────────
    if backprop_train and model_type == 'MLP':
        if start_this_W is not None:
            init_weights = [np.copy(w) for w in start_this_W]
        else:
            _w_pop_tmp, _ = define_genomes(
                input_shape=len(x_train[0]), output_shape=len(y_train[0]),
                neurons=neurons, activation_functions=activation_functions,
                population_size=1, dtype=dtype)
            init_weights = []
            for W in _w_pop_tmp[0]:
                fan_in  = W.shape[1]
                fan_out = W.shape[0]
                std = np.sqrt(2.0 / (fan_in + fan_out))
                init_weights.append(
                    np.random.normal(0, std, W.shape).astype(dtype)
                )
        return _run_backprop_train(init_weights, iter, activation_functions, None)
    # ─────────────────────────────────────────────────────────────────────────

    viz_objects = initialize_visualization_for_learner(show_history, neurons_history, neural_web_history, x_train, y_train)
    best_acc = 0
    best_loss = float('inf')
    best_fitness = float('-inf')
    best_acc_per_gen_list = []
    postfix_dict = {}
    loss_list = []
    target_pop = []

    progress = initialize_loading_bar(
        total= iter if isinstance(iter, int) else iter[0] + iter[1],
        desc="", ncols=85, bar_format=bar_format_learner)

    if fit_start is False:
        weight_pop, act_pop = define_genomes(
            input_shape=len(x_train[0]), output_shape=len(y_train[0]),
            neurons=neurons, activation_functions=activations,
            population_size=pop_size, dtype=dtype)
    else:
        weight_pop = [0] * len(activations)
        act_pop    = [0] * len(activations)

    if start_this_act is not None and start_this_W is not None:
        weight_pop[0] = start_this_W
        act_pop[0]    = start_this_act

    for i in range(iter):

       # PLAN → PTNN dönüşümü
        if model_type == 'PLAN' and transfer_learning:

            if i == iter_copy[0]:
                model_type = 'PTNN'
                neurons = neurons_copy

                for individual in range(len(weight_pop)):
                    weight_pop[individual] = np.copy(best_weight)
                    activation_potentiations[individual] = (
                        final_activations.copy()
                        if isinstance(final_activations, list) else final_activations)

                activation_potentiation = activation_potentiations[0]

                plan_W = best_weight
                extra_rows = neurons_copy[0] - len(y_train[0])
                fan_in  = plan_W.shape[1]
                fan_out = neurons_copy[0]

                if backprop_train:
                    std = np.sqrt(2.0 / (fan_in + fan_out))
                    extra_W = np.random.normal(0, std, (extra_rows, fan_in)).astype(dtype)

                else:
                    extra_W = np.zeros((extra_rows, fan_in), dtype=dtype)

                expanded_W = np.vstack([plan_W, extra_W])

                weight_pop, act_pop = define_genomes(
                    input_shape=len(x_train[0]), output_shape=len(y_train[0]),
                    neurons=neurons_copy, activation_functions=activation_functions,
                    population_size=pop_size, dtype=dtype)
                
                for l in range(len(weight_pop)):
                    weight_pop[l][0] = np.copy(expanded_W)

                for l in range(1, len(weight_pop[0])):
                    original_shape = weight_pop[0][l].shape
                    weight_pop[0][l] = np.eye(original_shape[0], original_shape[1],
                                            dtype=weight_pop[0][l].dtype)

                best_weight       = np.array(weight_pop[0], dtype=object)
                final_activations = act_pop[0]
                is_mlp     = True
                fit_start  = False

                # ── PTNN + backprop_train ─────────────────────────────────────
                if backprop_train:
                    progress.close()
                    return _run_backprop_train(
                        init_weights=list(best_weight),
                        grad_epochs= iter - i,
                        final_activations=final_activations,
                        activation_potentiation=activation_potentiation)
                # ─────────────────────────────────────────────────────────────

        postfix_dict["Gen"] = str(i + 1) + '/' + str(iter)
        progress.set_postfix(postfix_dict)

        if parallel_training:
            eval_params = []
            fit_params  = []

            with Pool(processes=thread_count) as pool:
                x_train_batch, y_train_batch = batcher(x_train, y_train, batch_size)

                if model_type == 'PLAN' and i == 0:
                    act_pop = copy.deepcopy(activations)

                eval_params = [
                    (x_train_batch, y_train_batch, None, model_type,
                     weight_pop[j], act_pop[j], activation_potentiations[j],
                     auto_normalization, None, cuda)
                    for j in range(pop_size)
                ]

                if model_type == 'PLAN':
                    if weight_evolve is True and i > 0:
                        pass
                    else:
                        fit_params = [
                            (x_train_batch, y_train_batch, act_pop[j],
                             weight_pop[j], auto_normalization, cuda, dtype)
                            for j in range(pop_size)
                        ]
                        W = pool.starmap(plan_fit, fit_params)
                        weight_pop  = W
                        eval_params = [(*ep[:4], w, *ep[5:])
                                       for ep, w in zip(eval_params, W)]

                start_W   = start_this_W   if (start_this_W   is not None and i == 0) else eval_params[0][4]
                start_act = start_this_act if (start_this_act is not None and i == 0) else eval_params[0][5]

                eval_params[0] = (
                    eval_params[0][0], eval_params[0][1], eval_params[0][2],
                    eval_params[0][3], start_W, start_act,
                    eval_params[0][6], eval_params[0][7],
                    eval_params[0][8], eval_params[0][9])

                models = pool.starmap(evaluate, eval_params)
                y_preds = [m[get_preds_softmax()] for m in models]
                loss_func = categorical_crossentropy if loss == 'categorical_crossentropy' else binary_crossentropy
                losses = pool.starmap(loss_func, [(eval_params[f][1], y_preds[f]) for f in range(pop_size)])
                target_pop = pool.starmap(wals, [(models[f][get_acc()], losses[f], acc_impact, loss_impact) for f in range(pop_size)])
                if cuda:
                    target_pop = [fit.get() for fit in target_pop]

            best_idx  = np.argmax(target_pop)
            best_fitness = target_pop[best_idx]
            best_weight  = models[best_idx][0]
            best_loss    = losses[best_idx]
            best_acc     = models[best_idx][get_acc()]
            best_model   = models[best_idx]
            x_train_batch = eval_params[best_idx][0]
            y_train_batch = eval_params[best_idx][1]

            if isinstance(eval_params[best_idx][5], list) and model_type == 'PLAN':
                final_activations = eval_params[best_idx][5].copy()
            elif isinstance(eval_params[best_idx][5], str):
                final_activations = eval_params[best_idx][5]
            else:
                final_activations = copy.deepcopy(eval_params[best_idx][5])

            if model_type == 'PLAN':
                final_activations = ([final_activations[0]]
                                     if len(set(final_activations)) == 1
                                     else final_activations)
            if batch_size == 1:
                postfix_dict[f"{data} Accuracy"] = format_number(best_acc)
                postfix_dict[f"{data} Loss"]     = format_number(best_loss)
                progress.set_postfix(postfix_dict)
            if show_current_activations:
                print(f", Current Activations={final_activations}", end='')

        else:
                
            for j in range(pop_size):
                x_train_batch, y_train_batch = batcher(x_train, y_train, batch_size=batch_size)

                if fit_start is True and i == 0:
                    if not (start_this_act is not None and j == 0):
                        act_pop[j] = activations[j]
                        W = plan_fit(x_train_batch, y_train_batch, activations=act_pop[j],
                                     cuda=cuda, auto_normalization=auto_normalization, dtype=dtype)
                        weight_pop[j] = W

                if weight_evolve is False:
                    weight_pop[j] = plan_fit(x_train_batch, y_train_batch,
                                             activations=act_pop[j], cuda=cuda,
                                             auto_normalization=auto_normalization, dtype=dtype)

                model = evaluate(x_train_batch, y_train_batch, W=weight_pop[j],
                                 activations=act_pop[j],
                                 activation_potentiations=activation_potentiations[j],
                                 auto_normalization=auto_normalization, cuda=cuda,
                                 model_type=model_type)
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
                    best_acc     = acc
                    best_loss    = train_loss
                    best_weight  = (np.copy(weight_pop[j]) if model_type == 'PLAN'
                                    else copy.deepcopy(weight_pop[j]))
                    best_model   = model

                    if isinstance(act_pop[j], list) and model_type == 'PLAN':
                        final_activations = act_pop[j].copy()
                    elif isinstance(act_pop[j], str):
                        final_activations = act_pop[j]
                    else:
                        final_activations = copy.deepcopy(act_pop[j])

                    if model_type == 'PLAN':
                        final_activations = ([final_activations[0]]
                                             if len(set(final_activations)) == 1
                                             else final_activations)
                    if batch_size == 1:
                        postfix_dict[f"{data} Accuracy"] = format_number(best_acc)
                        postfix_dict[f"{data} Loss"]     = format_number(best_loss)
                        progress.set_postfix(postfix_dict)
                    if show_current_activations:
                        print(f", Current Activations={final_activations}", end='')

                    if target_acc is not None and best_acc >= target_acc:
                        progress.close()
                        train_model = evaluate(x_train, y_train, W=best_weight,
                                               activations=final_activations,
                                               activation_potentiations=activation_potentiation,
                                               auto_normalization=auto_normalization,
                                               cuda=cuda, model_type=model_type)
                        if loss == 'categorical_crossentropy':
                            train_loss = categorical_crossentropy(y_true_batch=y_train, y_pred_batch=train_model[get_preds_softmax()])
                        else:
                            train_loss = binary_crossentropy(y_true_batch=y_train, y_pred_batch=train_model[get_preds_softmax()])
                        print('\nActivations: ', final_activations)
                        print('Activation Potentiation: ', activation_potentiation)
                        print('Train Accuracy:', train_model[get_acc()])
                        print('Train Loss: ', train_loss, '\n')
                        print('Model Type:', model_type)
                        if decision_boundary_history: display_decision_boundary_history(fig, artist, interval=interval)
                        display_visualizations_for_learner(viz_objects, best_weight, data, best_acc, best_loss, y_train, interval)
                        model = template_model._replace(weights=best_weight, predictions=best_model[get_preds()],
                                                       accuracy=best_acc, activations=final_activations,
                                                       softmax_predictions=best_model[get_preds_softmax()],
                                                       model_type=model_type, activation_potentiation=activation_potentiation)
                        return model

                    if target_loss is not None and best_loss <= target_loss:
                        progress.close()
                        train_model = evaluate(x_train, y_train, W=best_weight,
                                               activations=final_activations,
                                               activation_potentiations=activation_potentiation,
                                               auto_normalization=auto_normalization,
                                               cuda=cuda, model_type=model_type)
                        if loss == 'categorical_crossentropy':
                            train_loss = categorical_crossentropy(y_true_batch=y_train, y_pred_batch=train_model[get_preds_softmax()])
                        else:
                            train_loss = binary_crossentropy(y_true_batch=y_train, y_pred_batch=train_model[get_preds_softmax()])
                        print('\nActivations: ', final_activations)
                        print('Activation Potentiation: ', activation_potentiation)
                        print('Train Accuracy:', train_model[get_acc()])
                        print('Train Loss: ', train_loss, '\n')
                        print('Model Type:', model_type)
                        if decision_boundary_history: display_decision_boundary_history(fig, artist, interval=interval)
                        display_visualizations_for_learner(viz_objects, best_weight, data, best_acc, train_loss, y_train, interval)
                        model = template_model._replace(weights=best_weight, predictions=best_model[get_preds()],
                                                       accuracy=best_acc, activations=final_activations,
                                                       softmax_predictions=best_model[get_preds_softmax()],
                                                       model_type=model_type, activation_potentiation=activation_potentiation)

                        return model
                    
                    if model_type == 'PLAN' and transfer_learning and quick_start:
                        break
        
        progress.update(1)

        if batch_size != 1:
            train_model = evaluate(x_train, y_train, W=best_weight, activations=final_activations,
                                   activation_potentiations=activation_potentiation,
                                   auto_normalization=auto_normalization, cuda=cuda, model_type=model_type)
            if loss == 'categorical_crossentropy':
                train_loss = categorical_crossentropy(y_true_batch=y_train, y_pred_batch=train_model[get_preds_softmax()])
            else:
                train_loss = binary_crossentropy(y_true_batch=y_train, y_pred_batch=train_model[get_preds_softmax()])
            best_acc_per_gen_list.append(train_model[get_acc()])
            loss_list.append(train_loss)
            postfix_dict[f"{data} Accuracy"] = format_number(train_model[get_acc()])
            postfix_dict[f"{data} Loss"]     = format_number(train_loss)
            progress.set_postfix(postfix_dict)
        else:
            best_acc_per_gen_list.append(best_acc)
            loss_list.append(best_loss)

        if show_history:
            if batch_size == 1:
                gen_list = range(1, len(best_acc_per_gen_list) + 2)
                update_history_plots_for_learner(viz_objects, gen_list,
                    loss_list + [best_loss], best_acc_per_gen_list + [best_acc], x_train, final_activations)
            else:
                gen_list = range(1, len(best_acc_per_gen_list) + 2)
                update_history_plots_for_learner(viz_objects, gen_list,
                    loss_list + [loss_list[-1]], best_acc_per_gen_list + [best_acc_per_gen_list[-1]], x_train, final_activations)

        if neurons_history:
            viz_objects['neurons']['artists'] = update_neuron_history_for_learner(
                np.copy(best_weight), viz_objects['neurons']['ax'],
                viz_objects['neurons']['row'], viz_objects['neurons']['col'],
                y_train[0], viz_objects['neurons']['artists'],
                data=data, fig1=viz_objects['neurons']['fig'], acc=best_acc, loss=train_loss)

        if neural_web_history:
            art5_1, art5_2, art5_3 = draw_neural_web(W=best_weight, ax=viz_objects['web']['ax'],
                                                      G=viz_objects['web']['G'], return_objs=True)
            viz_objects['web']['artists'].append([art5_1] + [art5_2] + list(art5_3.values()))

        if decision_boundary_history:
            if i == 0:
                fig, ax = create_decision_boundary_hist()
                artist  = []
            artist = plot_decision_boundary(x_train_batch, y_train_batch,
                                            activations=act_pop[np.argmax(target_pop)],
                                            W=weight_pop[np.argmax(target_pop)],
                                            cuda=cuda, model_type=model_type, ax=ax, artist=artist)

        if model_type == 'PLAN' and transfer_learning and quick_start:
                    for z in range(len(weight_pop)):
                        weight_pop[z] = weight_pop[0]
        
                        
        weight_pop = (np.array(weight_pop, copy=False, dtype=dtype)
                    if model_type == 'PLAN'
                    else np.array(weight_pop, copy=False, dtype=object))


        if quick_start is True and transfer_learning is True:
            pass
        else:
            weight_pop, act_pop = genetic_optimizer(weight_pop, act_pop, i,
                                            np.array(target_pop, dtype=dtype, copy=False),
                                            weight_evolve=weight_evolve, is_mlp=is_mlp, bar_status=False)
        
        target_pop = []

        if early_stop and i > 0:
            if best_acc_per_gen_list[i] == best_acc_per_gen_list[i - 1]:
                progress.close()
                train_model = evaluate(x_train, y_train, W=best_weight,
                                       activations=final_activations,
                                       activation_potentiations=activation_potentiation,
                                       auto_normalization=auto_normalization,
                                       cuda=cuda, model_type=model_type)
                if loss == 'categorical_crossentropy':
                    train_loss = categorical_crossentropy(y_true_batch=y_train, y_pred_batch=train_model[get_preds_softmax()])
                else:
                    train_loss = binary_crossentropy(y_true_batch=y_train, y_pred_batch=train_model[get_preds_softmax()])
                print('\nActivations: ', final_activations)
                print('Activation Potentiation: ', activation_potentiation)
                print('Train Accuracy:', train_model[get_acc()])
                print('Train Loss: ', train_loss, '\n')
                print('Model Type:', model_type)
                if decision_boundary_history: display_decision_boundary_history(fig, artist, interval=interval)
                display_visualizations_for_learner(viz_objects, best_weight, data, best_acc, train_loss, y_train, interval)
                model = template_model._replace(weights=best_weight, predictions=best_model[get_preds()],
                                               accuracy=best_acc, activations=final_activations,
                                               softmax_predictions=best_model[get_preds_softmax()],
                                               model_type=model_type, activation_potentiation=activation_potentiation)
                return model

    # Final evaluation
    progress.close()
    train_model = evaluate(x_train, y_train, W=best_weight,
                           activations=final_activations,
                           activation_potentiations=activation_potentiation,
                           auto_normalization=auto_normalization, cuda=cuda, model_type=model_type)
    if loss == 'categorical_crossentropy':
        train_loss = categorical_crossentropy(y_true_batch=y_train, y_pred_batch=train_model[get_preds_softmax()])
    else:
        train_loss = binary_crossentropy(y_true_batch=y_train, y_pred_batch=train_model[get_preds_softmax()])

    print('\nActivations: ', final_activations)
    print('Activation Potentiation: ', activation_potentiation)
    print('Train Accuracy:', train_model[get_acc()])
    print('Train Loss: ', train_loss, '\n')
    print('Model Type:', model_type)
    if decision_boundary_history: display_decision_boundary_history(fig, artist, interval=interval)
    display_visualizations_for_learner(viz_objects, best_weight, data, best_acc, train_loss, y_train, interval)
    model = template_model._replace(weights=best_weight, predictions=best_model[get_preds()],
                                   accuracy=best_acc, activations=final_activations,
                                   softmax_predictions=best_model[get_preds_softmax()],
                                   model_type=model_type, activation_potentiation=activation_potentiation)
    
    return model


def grad(
    y_train:       np.ndarray,      # (N, C) one-hot
    softmax_preds: np.ndarray,      # (N, C) evaluate'den gelen softmax çıkışı
    cache:         list,            # predict_from_memory(return_activations=True) çıktısı
                                    # MLP: [Input, Z0, Z1, ..., Z_last]
    weights:       list,            # [W0, W1, …]  Wi.shape = (out_i, in_i)
    activations:   list,            # gizli katman aktivasyon isimleri ['tanh', 'relu', …]
                                    # len == num_layers - 1
    method:     str  = 'adam',      # 'sgd'|'momentum'|'rmsprop'|'adam'|'adamw'|'nadam'|'adagrad'
    state:         dict = None,     # optimizer state — ilk çağrıda None ver
    t:             int  = 1,        # epoch / zaman adımı
    learning_rate: float = 0.01,
    beta1:         float = 0.9,
    beta2:         float = 0.999,
    epsilon:       float = 1e-8,
    momentum:      float = 0.9,     # sgd-momentum / nesterov
    rho:           float = 0.9,     # rmsprop
    l2:            float = 0.0,     # L2 weight decay
    l1:            float = 0.0,     # L1 regularization
    clip_value:    float = 0.0,     # gradient clipping (0 = kapali)
    cuda:          bool  = False,
) -> tuple:
    """
    Performs a single backpropagation + optimizer step for MLP / PTNN models.
    The forward pass is NOT performed here — it must be done beforehand via
    evaluate(return_cache=True), whose outputs (cache and softmax_preds) are passed directly to this function.

    Supported optimizers: SGD with momentum, Nesterov momentum, RMSProp, Adam, AdamW, NAdam, Adagrad.

    This function is designed to be used as the gradient_optimizer argument in nn.learn().
    Use a lambda to pre-bind all hyperparameters and pass it to learn(). Example:
    ```python
    from pyerualjetwork.nn import grad

    gradient_optimizer = lambda *args, **kwargs: grad(*args,
                                                      method='adam',
                                                      learning_rate=0.001,
                                                      **kwargs)
    model = nn.learn(x_train, y_train, gradient_optimizer=gradient_optimizer,
                     iter=50, backprop_train=True,
                     neurons=[64, 64], activation_functions=['relu', 'relu'],
                     step_size=32)
    ```

    Examples:

        Adam (default):
        ```python
        gradient_optimizer = lambda *args, **kwargs: grad(*args,
                                                          method='adam',
                                                          learning_rate=0.001,
                                                          **kwargs)
        ```

        AdamW with L2 weight decay and gradient clipping:
        ```python
        gradient_optimizer = lambda *args, **kwargs: grad(*args,
                                                          method='adamw',
                                                          learning_rate=0.001,
                                                          l2=1e-4,
                                                          clip_value=1.0,
                                                          **kwargs)
        ```

        SGD with momentum:
        ```python
        gradient_optimizer = lambda *args, **kwargs: grad(*args,
                                                          method='sgd',
                                                          learning_rate=0.01,
                                                          momentum=0.9,
                                                          **kwargs)
        ```

        NAdam with L1 regularization:
        ```python
        gradient_optimizer = lambda *args, **kwargs: grad(*args,
                                                          method='nadam',
                                                          learning_rate=0.001,
                                                          l1=1e-5,
                                                          **kwargs)
        ```

    :Args:
    :param y_train: (np.ndarray): One-hot encoded ground truth labels. Shape: (N, C).
    :param softmax_preds: (np.ndarray): Softmax output from the forward pass via evaluate(return_cache=True). Shape: (N, C).
    :param cache: (list): Activation cache produced by evaluate(return_cache=True). Format — cache[0]: Input (N, in0), cache[i+1]: pre-activation output Z_i of layer i (N, out_i).
    :param weights: (list): List of weight matrices [W0, W1, ...] where Wi.shape = (out_i, in_i). Updated in-place and returned.
    :param activations: (list[str]): Activation function names for each hidden layer, e.g. ['relu', 'tanh']. Length must equal number of layers minus one.
    :param method: (str, optional): Optimizer algorithm. Options: 'sgd', 'momentum', 'rmsprop', 'adam', 'adamw', 'nadam', 'adagrad'. Default: 'adam'.
    :param state: (dict, optional): Optimizer state dictionary (running moments, accumulators etc.). Pass None on the first call — state will be initialized automatically and returned for reuse in subsequent steps. Default: None.
    :param t: (int, optional): Current time step / epoch index. Used for bias correction in Adam-family optimizers. Should be incremented each call. Default: 1.
    :param learning_rate: (float, optional): Step size for weight updates. Default: 0.01.
    :param beta1: (float, optional): Exponential decay rate for the first moment estimate (Adam / AdamW / NAdam). Default: 0.9.
    :param beta2: (float, optional): Exponential decay rate for the second moment estimate (Adam / AdamW / NAdam). Default: 0.999.
    :param epsilon: (float, optional): Small constant added to the denominator for numerical stability. Default: 1e-8.
    :param momentum: (float, optional): Momentum coefficient for 'sgd' and Nesterov 'momentum' optimizers. Default: 0.9.
    :param rho: (float, optional): Decay factor for the moving average of squared gradients in RMSProp. Default: 0.9.
    :param l2: (float, optional): L2 regularization (weight decay) coefficient. Applied directly in the gradient for all optimizers except 'adamw', where it is applied as a separate decoupled decay. Default: 0.0 (disabled).
    :param l1: (float, optional): L1 regularization coefficient. Adds sign(W) * l1 to the gradient. Default: 0.0 (disabled).
    :param clip_value: (float, optional): Element-wise gradient clipping threshold. Gradients are clipped to [-clip_value, clip_value] before the optimizer step. 0.0 disables clipping. Default: 0.0.
    :param cuda: (bool, optional): If True, uses CUDA-compatible activation derivative functions. Default: False.

    Returns:
        tuple: (weights, state) — updated weight matrices and the new optimizer state to be passed back on the next call.
    """

    from .cpu.activation_functions import apply_activation, apply_activation_derivative
 
    num_layers = len(weights)
    N          = y_train.shape[0]
    opt        = method.lower()

    # ── Acts: cache'den aktivasyon SONRASI değerler ───────────────────────────
    # cache[0]        = Input  (aktivasyon yok)
    # cache[i+1]      = Z_i    (pre-activation)  → apply_activation → A_i
    # acts[num_layers] = softmax_preds
    #
    # List comprehension: döngü Python overhead'ı minimum,
    # apply_activation BLAS-vektörize numpy işlemi.

    acts = (
        [cache[0]]
        + [apply_activation(cache[i + 1], [activations[i]]) for i in range(num_layers - 1)]
        + [softmax_preds]
    )

    # ── State başlatma ────────────────────────────────────────────────────────
    if state is None:
        state = {}
        if opt in ('adam', 'adamw', 'nadam'):
            state['m'] = [np.zeros_like(W) for W in weights]
            state['v'] = [np.zeros_like(W) for W in weights]
        elif opt in ('rmsprop', 'momentum', 'sgd'):
            state['v'] = [np.zeros_like(W) for W in weights]
        elif opt == 'adagrad':
            state['G'] = [np.zeros_like(W) for W in weights]

    # ── GERİ YAYILIM ─────────────────────────────────────────────────────────
    # Softmax + CCE birleşik türevi: delta = (ŷ − y) / N  (in-place)
    delta = softmax_preds - y_train          # yeni array (softmax_preds korunur)
    delta *= 1.0 / N                         # in-place bölme

    grads = [None] * num_layers

    for i in range(num_layers - 1, -1, -1):

        # dL/dW_i = delta.T @ acts[i]   (out_i, in_i)
        g = delta.T @ acts[i]

        # Regularization
        if l2 > 0.0 and opt != 'adamw':
            g += l2 * weights[i]             # in-place
        if l1 > 0.0:
            g += l1 * np.sign(weights[i])    # in-place

        # Gradient clipping (element-wise, in-place)
        if clip_value > 0.0:
            np.clip(g, -clip_value, clip_value, out=g)

        grads[i] = g

        if i > 0:
            # delta → bir alt katmana ilet, in-place ReLU türevi
            # cache[i] = Z_{i-1}  (pre-activation of layer i-1)
            new_delta = delta @ weights[i]                               # (N, out_{i-1})
            dA_dZ = apply_activation_derivative(cache[i], activations[i - 1])
            np.multiply(new_delta, dA_dZ, out=new_delta)                 # in-place türev
            delta = new_delta

    # ── OPTİMİZER GÜNCELLEMESİ ───────────────────────────────────────────────

    if opt == 'sgd':
        for i in range(num_layers):
            # v = momentum*v + g  (in-place)
            np.multiply(momentum, state['v'][i], out=state['v'][i])
            np.add(state['v'][i], grads[i], out=state['v'][i])
            weights[i] -= learning_rate * state['v'][i]

    elif opt == 'momentum':
        # Nesterov: w ← w − momentum*v_prev + (1+momentum)*v_new
        for i in range(num_layers):
            v_prev = state['v'][i].copy()
            np.multiply(momentum, state['v'][i], out=state['v'][i])
            state['v'][i] -= learning_rate * grads[i]                   # in-place
            weights[i] += -momentum * v_prev + (1.0 + momentum) * state['v'][i]

    elif opt == 'rmsprop':
        for i in range(num_layers):
            # E[g²] = rho*E[g²] + (1-rho)*g²  (in-place)
            np.multiply(rho, state['v'][i], out=state['v'][i])
            state['v'][i] += (1.0 - rho) * (grads[i] * grads[i])
            denom = np.sqrt(state['v'][i])
            denom += epsilon                                             # in-place +ε
            weights[i] -= learning_rate * grads[i] / denom

    elif opt == 'adam':
        # Bias correction dışarıda tek hesap → her katmanda tekrar hesaplanmaz
        b1t    = 1.0 - beta1 ** t
        b2t    = 1.0 - beta2 ** t
        lr_adj = learning_rate * (b2t ** 0.5) / b1t   # combined lr correction
        for i in range(num_layers):
            # m = beta1*m + (1-beta1)*g  (in-place)
            np.multiply(beta1, state['m'][i], out=state['m'][i])
            state['m'][i] += (1.0 - beta1) * grads[i]
            # v = beta2*v + (1-beta2)*g²  (in-place)
            np.multiply(beta2, state['v'][i], out=state['v'][i])
            state['v'][i] += (1.0 - beta2) * (grads[i] * grads[i])
            # w -= lr_adj * m / (√v + ε)
            denom = np.sqrt(state['v'][i])
            denom += epsilon
            weights[i] -= lr_adj * state['m'][i] / denom

    elif opt == 'adamw':
        # Adam + ayrık weight decay
        b1t    = 1.0 - beta1 ** t
        b2t    = 1.0 - beta2 ** t
        lr_adj = learning_rate * (b2t ** 0.5) / b1t
        for i in range(num_layers):
            np.multiply(beta1, state['m'][i], out=state['m'][i])
            state['m'][i] += (1.0 - beta1) * grads[i]
            np.multiply(beta2, state['v'][i], out=state['v'][i])
            state['v'][i] += (1.0 - beta2) * (grads[i] * grads[i])
            denom = np.sqrt(state['v'][i])
            denom += epsilon
            weights[i] -= lr_adj * state['m'][i] / denom
            weights[i] -= learning_rate * l2 * weights[i]               # ayrık weight decay

    elif opt == 'nadam':
        # Nesterov Adam
        b1t  = 1.0 - beta1 ** t
        b1t1 = 1.0 - beta1 ** (t + 1)
        b2t  = 1.0 - beta2 ** t
        for i in range(num_layers):
            np.multiply(beta1, state['m'][i], out=state['m'][i])
            state['m'][i] += (1.0 - beta1) * grads[i]
            np.multiply(beta2, state['v'][i], out=state['v'][i])
            state['v'][i] += (1.0 - beta2) * (grads[i] * grads[i])
            # Nesterov m_hat: bir adım ileriden
            m_hat = beta1 * state['m'][i] / b1t1 + (1.0 - beta1) * grads[i] / b1t
            denom = np.sqrt(state['v'][i] / b2t)
            denom += epsilon
            weights[i] -= learning_rate * m_hat / denom

    elif opt == 'adagrad':
        for i in range(num_layers):
            state['G'][i] += grads[i] * grads[i]                        # in-place birikimli
            denom = np.sqrt(state['G'][i])
            denom += epsilon
            weights[i] -= learning_rate * grads[i] / denom

    else:
        raise ValueError(
            f"Unknown method: '{method}'. "
            "Options: 'sgd', 'momentum', 'rmsprop', 'adam', 'adamw', 'nadam', 'adagrad'"
        )

    return weights, state

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
    cuda=False,
    return_cache=False,         # True ise backprop için layer cache döner
) -> tuple:
    """
    Evaluates the neural network model using the given test data.
    Args:
        x_test (np.ndarray): Test data.
        y_test (np.ndarray): Test labels (one-hot encoded).
        model (tuple, optional): Trained model.
        model_type: (str, optional): Type of the model. Options: 'PLAN', 'MLP', 'PTNN'.
        W (array-like, optional): Neural net weight matrix.
        activations (list, optional): Activation list for PLAN or MLP models. Default = ['linear'].
        activation_potentiations (list, optional): Extra activation potentiation list for PTNN. Default = [].
        auto_normalization (bool, optional): Normalization for x_test? Default = False.
        show_report (bool, optional): Show test report. Default = False.
        cuda (bool, optional): CUDA GPU acceleration? Default = False.
        return_cache (bool, optional): If True, returns layer-wise pre-activation cache
                                       from predict_from_memory for use in backprop.
                                       Cache = [Input, Z0, Z1, ..., Z_last]
                                       Default = False.
    Returns:
        return_cache=False: W, preds, accuracy, None, None, softmax_preds
        return_cache=True : W, preds, accuracy, None, None, softmax_preds, cache
    """
    from .cpu.visualizations import plot_evaluate
    from .model_ops import predict_from_memory

    if not cuda:
        from .cpu.data_ops import normalization
        array_type = np

        if model:
            sample_acc = model.accuracy
            if hasattr(sample_acc, "get"): model = model._replace(accuracy=sample_acc.get())
    else:
        from .cuda.data_ops import normalization
        import cupy as cp
        array_type = cp
        x_test = cp.array(x_test)
        y_test = cp.array(y_test)
        if model:
            sample_acc = model.accuracy
            if isinstance(sample_acc, np.number): model = model._replace(accuracy=cp.array(sample_acc))

    if model is None:
        from .model_ops import get_model_template
        template_model = get_model_template()

        model = template_model._replace(
            weights=W,
            activations=activations,
            model_type=model_type,
            activation_potentiation=activation_potentiations
        )

    if return_cache:
        # predict_from_memory'den layer cache'i al
        # activations_list = [Input, Z0, Z1, ..., Z_last]  (pre-activation değerler)
        result, cache = predict_from_memory(x_test, model, cuda=cuda, return_activations=True)
    else:
        result = predict_from_memory(x_test, model, cuda=cuda)
        cache = None

    if auto_normalization:
        x_test = normalization(x_test, dtype=x_test.dtype)

    max_vals     = array_type.max(result, axis=1, keepdims=True)
    exp_shifted  = array_type.exp(result - max_vals)
    softmax_preds = exp_shifted / array_type.sum(exp_shifted, axis=1, keepdims=True)
    accuracy     = (array_type.argmax(softmax_preds, axis=1) == array_type.argmax(y_test, axis=1)).mean()

    if show_report:
        plot_evaluate(x_test, y_test, result, acc=accuracy, model=model, cuda=cuda)

    if return_cache:
        return W, result, accuracy, None, None, softmax_preds, cache

    return W, result, accuracy, None, None, softmax_preds