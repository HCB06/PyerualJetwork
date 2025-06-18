""" 


Model Operations
=======================
This module hosts functions for handling all operational processes related to models, including:

- Saving and loading models
- Making predictions from memory
- Making predictions from storage
- Retrieving model weights
- Retrieving model activation functions
- Retrieving model accuracy
- Running the model in reverse (applicable to PLAN models)

Module functions:
-----------------
- get_model_template()
- build_model()
- save_model()
- load_model()
- predict_from_storage()
- predict_from_memory()
- reverse_predict_from_storage()
- reverse_predict_from_memory()

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
from colorama import Fore, Style
from datetime import datetime
import pickle
from scipy import io
import scipy.io as sio
import pandas as pd
from collections import namedtuple

def get_model_template():
    """
    Creates and returns a named tuple template for a model structure.

    This function defines a `Model` named tuple with standard fields used
    to represent various components of a machine learning model.
    All fields are initialized to `None`.

    Returns:
        Model: A named tuple with the following fields initialized to None:
            -> weights: Model weights.
            -> predictions: Raw class predictions.
            -> accuracy: Model evaluation metric (e.g., accuracy).
            -> activations: Activation functions used in the model.
            -> scaler_params: Parameters used for feature scaling.
            -> softmax_predictions: Output probabilities after softmax.
            -> model_type: A string indicating the type of model.
            -> weights_type: The format or data type of the weights.
            -> weights_format: Structural format of the weights.
            -> model_version: A string or identifier for versioning.
            -> model_df: DataFrame holding model-related data.
            -> activation_potentiation: Only for PTNN models.
    """
    Model = namedtuple("Model", [
        "weights",
        "predictions",
        "accuracy",
        "activations",
        "scaler_params",
        "softmax_predictions",
        "model_type",
        "weights_type",
        "weights_format",
        "model_version",
        "model_df",
        "activation_potentiation"
    ])

    template_model = Model(None, None, None, None, None, None, None, None, None, None, None, None)

    return template_model


def build_model(W, activations, model_type, activation_potentiation=None):
    """
    Builds a model using the template and specified components.

    Args:
        W (any): The weight parameters of the model.
        activations (any): The activation functions to be used.
        model_type (str): A string specifying the model architecture (PLAN, MLP or PTNN).
        activation_potentiation (optional): An optional parameter only for PTNN models.

    Returns:
        Model: A named tuple representing the constructed model, with relevant fields filled in.
    """
    template_model = get_model_template()
    model = template_model._replace(
        weights=W,
        activations=activations,
        model_type=model_type,
        activation_potentiation=activation_potentiation
    )

    return model


def save_model(model,
               model_name,
               model_path='',
               weights_type='npy',
               weights_format='raw',
               show_architecture=False,
               show_info=True
               ):

    """
    Function to save a potentiation learning artificial neural network model.
    Args:

        model (tuple): Trained model.

        model_name: (str): Name of the model.
        
        model_path: (str): Path where the model will be saved. For example: C:/Users/beydili/Desktop/denemePLAN/ default: ''

        weights_type: (str): Type of weights to save (options: 'txt', 'pkl', 'npy', 'mat'). default: 'npy'
        
        weights_format: (str): Format of the weights (options: 'f', 'raw'). default: 'raw'
        
        show_architecture: (bool): It draws model architecture. True or False. Default: False. NOTE! draw architecture only works for PLAN models. Not MLP models for now, but it will be.
        
        show_info: (bool): Prints model details into console. default: True

    Returns:
        No return.
    """
    
    from .cpu.visualizations import draw_model_architecture
    from . import __version__

    model_type = model.model_type
    activations = model.activations
    activation_potentiation = model.activation_potentiation
    scaler_params = model.scaler_params
    W = model.weights
    acc = model.accuracy

    if model_type != 'PLAN' and model_type != 'MLP' and model_type != 'PTNN':
        raise ValueError("model_type parameter must be 'PLAN', 'MLP' or 'PTNN'.")

    if model_type == 'PTNN' and activation_potentiation == []:
        raise ValueError('PTNN models need extra activation_potentiation parameter.')

    if isinstance(activations, str):
        activations = [activations]
    else:
        activations = [item if isinstance(item, list) else [item] for item in activations]

    activations = activations.copy()

    if model_type == 'PTNN':
        if isinstance(activation_potentiation, str):
            activation_potentiation = [activation_potentiation]
        else:
            activation_potentiation = [item if isinstance(item, list) else [item] for item in activation_potentiation]

        activation_potentiation = activation_potentiation.copy()

    if acc != None:
        acc= float(acc)

    if weights_type != 'txt' and weights_type != 'npy' and weights_type != 'mat' and weights_type != 'pkl':
        raise ValueError(Fore.RED + "ERROR: Save Weight type (File Extension) Type must be 'txt' or 'npy' or 'mat' or 'pkl' from: save_model" + Style.RESET_ALL)

    if weights_format != 'd' and weights_format != 'f' and weights_format != 'raw':
        raise ValueError(Fore.RED + "ERROR: Weight Format Type must be 'd' or 'f' or 'raw' from: save_model" + Style.RESET_ALL)
        
    NeuronCount = []
    SynapseCount = []

    if model_type == 'PLAN':
        class_count = W.shape[0]

        try:
            NeuronCount.append(np.shape(W)[1])
            NeuronCount.append(np.shape(W)[0])
            SynapseCount.append(np.shape(W)[0] * np.shape(W)[1])
        except AttributeError as e:
            raise AttributeError(Fore.RED + "ERROR: W does not have a shape attribute. Check if W is a valid matrix." + Style.RESET_ALL) from e
        except IndexError as e:
            raise IndexError(Fore.RED + "ERROR: W has an unexpected shape format. Ensure it has two dimensions." + Style.RESET_ALL) from e
        except (TypeError, ValueError) as e:
            raise TypeError(Fore.RED + "ERROR: W is not a valid numeric matrix." + Style.RESET_ALL) from e
        except Exception as e:
            raise RuntimeError(Fore.RED + f"ERROR: An unexpected error occurred in save_model: {e}" + Style.RESET_ALL) from e

    elif model_type == 'MLP' or model_type == 'PTNN':

        class_count = W[-1].shape[0]
        
        NeuronCount.append(np.shape(W[0])[1])

        for i in range(len(W)):
            try:
                    NeuronCount.append(np.shape(W[i])[0])
                    SynapseCount.append(np.shape(W[i])[0] * np.shape(W[i])[1])
            except AttributeError as e:
                raise AttributeError(Fore.RED + "ERROR: W does not have a shape attribute. Check if W is a valid matrix." + Style.RESET_ALL) from e
            except IndexError as e:
                raise IndexError(Fore.RED + "ERROR: W has an unexpected shape format. Ensure it has two dimensions." + Style.RESET_ALL) from e
            except (TypeError, ValueError) as e:
                raise TypeError(Fore.RED + "ERROR: W is not a valid numeric matrix." + Style.RESET_ALL) from e
            except Exception as e:
                raise RuntimeError(Fore.RED + f"ERROR: An unexpected error occurred in save_model: {e}" + Style.RESET_ALL) from e

    
        SynapseCount.append(' ')
        
        activations.append('')
        activations.insert(0, '')

    if len(activations) == 1 and model_type == 'PLAN':
        activations = [activations]
        activations.append('')

    if model_type == 'PTNN':
        if len(activations) > len(activation_potentiation):
            for i in range(len(activations) - len(activation_potentiation)):
                activation_potentiation.append('')

        if len(activation_potentiation) > len(activations):
            for i in range(len(activation_potentiation) - len(activations)):
                activations.append('')

    if len(activations) > len(NeuronCount):
        for i in range(len(activations) - len(NeuronCount)):
            NeuronCount.append('')
        
    if len(activations) > len(SynapseCount):
        for i in range(len(activations) - len(SynapseCount)):
            SynapseCount.append('')

    if scaler_params != None:

        if len(scaler_params) > len(activations):

            activations += ['']

        elif len(activations) > len(scaler_params):

            for i in range(len(activations) - len(scaler_params)):

                scaler_params.append(' ')

    data = {'MODEL NAME': model_name,
            'MODEL TYPE': model_type,
            'CLASS COUNT': class_count,
            'NEURON COUNT': NeuronCount,
            'SYNAPSE COUNT': SynapseCount,
            'VERSION': __version__,
            'ACCURACY': acc,
            'SAVE DATE': datetime.now(),
            'WEIGHTS TYPE': weights_type,
            'WEIGHTS FORMAT': weights_format,
            'STANDARD SCALER': scaler_params,
            'ACTIVATION FUNCTIONS': activations,
            'ACTIVATION POTENTIATION': activation_potentiation
            }

    df = pd.DataFrame(data)
    df.to_pickle(model_path + model_name + '.pkl')

    try:

        if weights_type == 'txt' and weights_format == 'f':

                np.savetxt(model_path + model_name + f'_weights.txt',  W, fmt='%f')

        if weights_type == 'txt' and weights_format == 'raw':

                np.savetxt(model_path + model_name + f'_weights.txt',  W)

        ###

        
        if weights_type == 'pkl' and weights_format == 'f':

            with open(model_path + model_name + f'_weights.pkl', 'wb') as f:
                pickle.dump(W.astype(float), f)

        if weights_type == 'pkl' and weights_format =='raw':
        
            with open(model_path + model_name + f'_weights.pkl', 'wb') as f:
                pickle.dump(W, f)

        ###

        if weights_type == 'npy' and weights_format == 'f':

                np.save(model_path + model_name + f'_weights.npy',  W, W.astype(float))

        if weights_type == 'npy' and weights_format == 'raw':

                np.save(model_path + model_name + f'_weights.npy',  W)

        ###

        if weights_type == 'mat' and weights_format == 'f':

                w = {'w': W.astype(float)}
                io.savemat(model_path + model_name + f'_weights.mat', w)

        if weights_type == 'mat' and weights_format == 'raw':
                
                w = {'w': W}
                io.savemat(model_path + model_name + f'_weights.mat', w)


    except OSError as e:
        raise OSError(Fore.RED + f"ERROR: An OSError error occurred in save_model at saving weights. Maybe model name or path or administration issue: {e}" + Style.RESET_ALL) from e

    if show_info:
        print(df)
    
        message = (
            Fore.GREEN + "Model Saved Successfully\n" +
            Fore.MAGENTA + "Don't forget, if you want to load model: model log file and weight files must be in the same directory." +
            Style.RESET_ALL
        )
        
        print(message)

    if show_architecture:
        draw_model_architecture(model_name=model_name, model_path=model_path)



def load_model(model_name,
               model_path,
               ):
    """
   Function to load a potentiation learning model.

   Args:
    model_name (str): Name of the model.
    
    model_path (str): Path where the model is saved.

   Returns:
    tuple(model): Weights, None, accuracy, activations, scaler_params, None, model_type, weight_type, weight_format, device_version, (list[df_elements])=Pandas DataFrame of the model, activation_potentiation
    """

    from . import __version__

    try:

         df = pd.read_pickle(model_path + model_name + '.pkl')

    except OSError as e:
        raise OSError(Fore.RED + f"ERROR: An OSError error occurred in load_model at loading model params. Maybe model name or path or administration issue: {e}" + Style.RESET_ALL) from e



    scaler_params = df['STANDARD SCALER'].tolist()
    
    try:
        if scaler_params[0] == None:
            scaler_params = scaler_params[0]

    except:
        scaler_params = [item for item in scaler_params if isinstance(item, np.ndarray)]

     
    model_name = str(df['MODEL NAME'].iloc[0])
    model_type = str(df['MODEL TYPE'].iloc[0])
    WeightType = str(df['WEIGHTS TYPE'].iloc[0])
    WeightFormat = str(df['WEIGHTS FORMAT'].iloc[0])
    acc = str(df['ACCURACY'].iloc[0])

    activations = list(df['ACTIVATION FUNCTIONS'])
    activations = [x for x in activations if not (isinstance(x, float) and np.isnan(x))]
    activations = [item for item in activations if item != ''] 

    activation_potentiation = list(df['ACTIVATION POTENTIATION'])
    activation_potentiation = [x for x in activation_potentiation if not (isinstance(x, float) and np.isnan(x))]
    activation_potentiation = [item for item in activation_potentiation if item != ''] 

    device_version = __version__

    try:
        model_version = str(df['VERSION'].iloc[0])
        if model_version != device_version:
            message = (
            Fore.MAGENTA + f"WARNING: Your PyerualJetwork version({device_version}) is different from this model's version({model_version}).\nIf you have a performance issue, please install this model version. Use this: pip install pyerualjetwork=={model_version} or look issue_solver module." +
            Style.RESET_ALL
        )
            print(message)
        
    except:
        pass # Version check only in >= 5.0.2

    if model_type == 'MLP' or model_type == 'PTNN': allow_pickle = True
    else: allow_pickle = False

    try:
        if WeightType == 'txt':
                W = (np.loadtxt(model_path + model_name + f'_weights.txt'))
        elif WeightType == 'npy':
                W = (np.load(model_path + model_name + f'_weights.npy', allow_pickle=allow_pickle))
        elif WeightType == 'mat':
                W = (sio.loadmat(model_path + model_name + f'_weights.mat'))
        elif WeightType == 'pkl':
            with open(model_path + model_name + f'_weights.pkl', 'rb') as f:
                W = pickle.load(f)
        else:

            raise ValueError(
                Fore.RED + "Incorrect weight type value. Value must be 'txt', 'npy', 'pkl' or 'mat' from: load_model." + Style.RESET_ALL)

    except OSError as e:
        raise OSError(Fore.RED + f"ERROR: An OSError error occurred in load_model at loading weights. Maybe model name or path or administration issue: {e}" + Style.RESET_ALL) from e


    if WeightType == 'mat':
        W = W['w']

    template_model = get_model_template()

    model = template_model._replace(weights=W,
                                    accuracy=acc,
                                    activations=activations,
                                    scaler_params=scaler_params,
                                    weights_type=WeightType,
                                    weights_format=WeightFormat,
                                    model_version=device_version,
                                    model_df=df,
                                    model_type=model_type,
                                    activation_potentiation=activation_potentiation)

    return model



def predict_from_storage(Input, model_name, cuda=False, model_path=''):

    """
    Function to make a prediction
    from storage

    Args:
        Input (list or ndarray): Input data for the model.
        
        model_name (str): Name of the model.
        
        cuda (bool, optional): CUDA GPU acceleration ? Default = False.
        
        model_path (str): Path of the model. Default: ''
    
    Returns:
        ndarray: Output from the model.
    """
    
    if cuda:
        if not isinstance(Input, cp.ndarray): Input = cp.array(Input)
        from .cuda.activation_functions import apply_activation
        from .cuda.data_ops import standard_scaler
    
    else:
        from .cpu.activation_functions import apply_activation
        from .cpu.data_ops import standard_scaler

    try:

        model = load_model(model_name, model_path)
        
        model_type = model.model_type
        activations = model.activations
        activation_potentiation = model.activation_potentiation
        scaler_params = model.scaler_params
        W = model.weights

        if cuda and scaler_params is not None:
            if not isinstance(scaler_params[0], cp.ndarray): scaler_params[0] = cp.array(scaler_params[0])
            if not isinstance(scaler_params[1], cp.ndarray): scaler_params[1] = cp.array(scaler_params[1])

        Input = standard_scaler(None, Input, scaler_params)

        if isinstance(activations, str):
            activations = [activations]
        elif isinstance(activations, list):
            activations = [item if isinstance(item, list) or isinstance(item, str) else [item] for item in activations]

        if model_type == 'MLP':
            
            layer = Input @ cp.array(W[0]).T if cuda else Input @ W[0].T
            for i in range(1, len(W)):
                    layer = apply_activation(layer, activations[i-1])
                    layer = layer @ cp.array(W[i]).T if cuda else layer @ W[i].T

            result = layer

        if model_type == 'PLAN':

            Input = apply_activation(Input, activations)
            result = Input @ cp.array(W).T if cuda else Input @ W.T
        
        if model_type == 'PTNN':

            if isinstance(activation_potentiation, str):
                activation_potentiation = [activation_potentiation]
            elif isinstance(activation_potentiation, list):
                activation_potentiation = [item if isinstance(item, list) or isinstance(item, str) else [item] for item in activation_potentiation]

            Input = apply_activation(Input, activation_potentiation)
            layer = Input @ cp.array(W[0]).T if cuda else Input @ W[0].T

            for i in range(1, len(W)):
                layer = apply_activation(layer, activations[i-1])
                layer = layer @ cp.array(W[i]).T if cuda else layer @ W[i].T

            result = layer

        return result
    
    except Exception as e:
        raise RuntimeError(Fore.RED + f"ERROR: An error occurred in predict_from_storage {e}" + Style.RESET_ALL) from e


def reverse_predict_from_storage(output, model_name, cuda=False, model_path=''):

    """
    reverse prediction function from storage
    Args:

        output (list or ndarray): output layer for the model .

        model_name (str): Name of the model.

        cuda (bool, optional): CUDA GPU acceleration ? Default = False.
        
        model_path (str): Path of the model. Default: ''

    Returns:
        ndarray: Input from the model.
    """

    if cuda:
        if not isinstance(output, cp.ndarray): output = cp.array(output)

    model = load_model(model_name, model_path)
    
    W = model.weights if not cuda else cp.array(model.weights)

    try:
        Input = W.T @ output
        return Input
    except Exception as e:
        raise RuntimeError(Fore.RED + f"ERROR: An error occurred {e}" + Style.RESET_ALL) from e
    


def predict_from_memory(Input, model, cuda=False):

    """
    Function to make a prediction.
    from memory.

    Args:
        Input (list or ndarray): Input data for the model.

        model (tuple): Trained model.

        cuda (bool, optional): CUDA GPU acceleration ? Default = False.
        
    Returns:
    ndarray: Output from the model.
    """
    
    model_type = model.model_type
    activations = model.activations
    activation_potentiation = model.activation_potentiation
    scaler_params = model.scaler_params
    W = model.weights

    if not cuda: 
        from .cpu.data_ops import standard_scaler
        from .cpu.activation_functions import apply_activation
    else:
        if scaler_params is not None:
            if not isinstance(scaler_params[0], cp.ndarray): scaler_params[0] = cp.array(scaler_params[0])
            if not isinstance(scaler_params[1], cp.ndarray): scaler_params[1] = cp.array(scaler_params[1])

        if not isinstance(Input, cp.ndarray): Input = cp.array(Input)
        from .cuda.data_ops import standard_scaler
        from .cuda.activation_functions import apply_activation

    if model_type != 'PLAN' and model_type != 'MLP' and model_type != 'PTNN': raise ValueError("model_type parameter must be 'PLAN', 'MLP' or 'PTNN'.")

    try:
    
        Input = standard_scaler(None, Input, scaler_params)
        
        if isinstance(activations, str):
            activations = [activations]
        elif isinstance(activations, list):
            activations = [item if isinstance(item, list) or isinstance(item, str) else [item] for item in activations]

        if model_type == 'MLP':
            
            layer = Input @ cp.array(W[0]).T if cuda else Input @ W[0].T
            for i in range(1, len(W)):
                    layer = apply_activation(layer, activations[i-1])
                    layer = layer @ cp.array(W[i]).T if cuda else layer @ W[i].T

            result = layer

        if model_type == 'PLAN':

            Input = apply_activation(Input, activations)
            result = Input @ cp.array(W).T if cuda else Input @ W.T
        
        if model_type == 'PTNN':

            if isinstance(activation_potentiation, str):
                activation_potentiation = [activation_potentiation]
            elif isinstance(activation_potentiation, list):
                activation_potentiation = [item if isinstance(item, list) or isinstance(item, str) else [item] for item in activation_potentiation]

            Input = apply_activation(Input, activation_potentiation)
            layer = Input @ cp.array(W[0]).T if cuda else Input @ W[0].T

            for i in range(1, len(W)):
                layer = apply_activation(layer, activations[i-1])
                layer = layer @ cp.array(W[i]).T if cuda else layer @ W[i].T

            result = layer
        
        return result

    except Exception as e:
        raise RuntimeError(Fore.RED + f"ERROR: An error occurred in predict_from_memory {e}" + Style.RESET_ALL) from e

def reverse_predict_from_memory(output, W, cuda=False):

    """
    reverse prediction function from memory

    Args:

        output (list or ndarray): output layer for the model.

        W (ndarray): Weights of the model.

        cuda (bool, optional): CUDA GPU acceleration ? Default = False.

    Returns:
        ndarray: Input from the model.
    """

    try:
        if cuda: W = cp.array(W)
        Input = W.T @ output
        return Input
    
    except Exception as e:
        raise RuntimeError(Fore.RED + f"ERROR: An error occurred {e}" + Style.RESET_ALL) from e


def get_weights():

    return 0


def get_preds():

    return 1


def get_acc():

    return 2


def get_act():

    return 3


def get_scaler():

    return 4


def get_preds_softmax():

    return 5


def get_model_type():

    return 6


def get_weights_type():
     
    return 7


def get_weights_format():
     
    return 8


def get_model_version():
     
    return 9


def get_model_df():
     
    return 10


def get_act_pot():
     
    return 11