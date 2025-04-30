""" 


Model Operations on CUDA
========================
This module hosts functions for handling all operational processes related to models on CUDA GPU, including:

- Saving and loading models
- Making predictions from memory
- Making predictions from storage
- Retrieving model weights
- Retrieving model activation functions
- Retrieving model accuracy
- Running the model in reverse (applicable to PLAN models)

Module functions:
-----------------
- save_model()
- load_model()
- predict_from_storage()
- predict_from_memory()
- reverse_predict_from_storage()
- reverse_predict_from_memory()
- get_weights()
- get_act()
- get_preds()
- get_preds_softmax()
- get_acc()
- get_scaler()
- get_model_type()
- get_weights_type()
- get_weights_format()
- get_model_version()
- get_model_df()
- get_act_pot()


Examples: https://github.com/HCB06/PyerualJetwork/tree/main/Welcome_to_PyerualJetwork/ExampleCodes

PyerualJetwork document: https://github.com/HCB06/PyerualJetwork/blob/main/Welcome_to_PyerualJetwork/PYERUALJETWORK_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf

- Creator: Hasan Can Beydili
- YouTube: https://www.youtube.com/@HasanCanBeydili
- Linkedin: https://www.linkedin.com/in/hasan-can-beydili-77a1b9270/
- Instagram: https://www.instagram.com/canbeydilj
- Contact: tchasancan@gmail.com
"""

import cupy as cp
import numpy as np
from colorama import Fore, Style
from datetime import datetime
import pickle
from scipy import io
import scipy.io as sio
import pandas as pd
import gc

def save_model(model_name,
               W,
               model_type,
               scaler_params=None,
               test_acc=None,
               model_path='',
               activations=['linear'],
               activation_potentiation=None,
               weights_type='npy',
               weights_format='raw',
               show_architecture=False,
               show_info=True
               ):

    """

    Function to save a potentiation learning artificial neural network model or trained non-bias MLP model.
    Args:
        model_name: (str): Name of the model.
        W: Weights of the model.

        model_type: (str): Type of the model. Options: 'PLAN', 'MLP', 'PTNN'.
        
        scaler_params: (list[num, num]): standard scaler params list: mean,std. If not used standard scaler then be: None.
        
        model_type: (str): Type of the model. default: 'PLAN'
        
        test_acc: (float): Test accuracy of the model. default: None
        
        model_path: (str): Path where the model will be saved. For example: C:/Users/beydili/Desktop/denemePLAN/ default: ''
       
        activations: (list[str]): For deeper PLAN networks, activation function parameters. Or activation function parameters for MLP layers. For more information please run this code: plan.activations_list() default: ['linear']
        
        activation_potentiation (list, optional): Extra activation potentiation list (PLAN layers activations) for just PTNN models. Default = None.

        weights_type: (str): Type of weights to save (options: 'txt', 'pkl', 'npy', 'mat'). default: 'npy'
        
        weights_format: (str): Format of the weights (options: 'f', 'raw'). default: 'raw'
        
        show_architecture: (bool): It draws model architecture. True or False. Default: False. NOTE! draw architecture only works for PLAN models. Not MLP models for now, but it will be.
        
        show_info: (bool): Prints model details into console. default: True


    Returns:
        No return.
    """
    
    from .visualizations import draw_model_architecture
    from .. import __version__

    if model_type != 'PLAN' and model_type != 'MLP' and model_type != 'PTNN':
        raise ValueError("model_type parameter must be 'PLAN', 'MLP' or 'PTNN'.")

    if model_type == 'PTNN' and activation_potentiation == []:
        raise ValueError('PTNN models need extra activation_potentiation parameter.')
    
    if isinstance(W, list):
        W = W.copy()

    if isinstance(W, np.ndarray):
        W = np.copy(W)
        
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

    if test_acc is not None:
        test_acc= float(test_acc)

    if weights_type != 'txt' and weights_type != 'npy' and weights_type != 'mat' and weights_type != 'pkl':
        raise ValueError(Fore.RED + "ERROR: Save Weight type (File Extension) Type must be 'txt' or 'npy' or 'mat' or 'pkl' from: save_model" + Style.RESET_ALL)

    if weights_format != 'd' and weights_format != 'f' and weights_format != 'raw':
        raise ValueError(Fore.RED + "ERROR: Weight Format Type must be 'd' or 'f' or 'raw' from: save_model" + Style.RESET_ALL)
        
    NeuronCount = []
    SynapseCount = []

    if model_type == 'PLAN':
        class_count = W.shape[0]

        try:
            NeuronCount.append(cp.shape(W)[1])
            NeuronCount.append(cp.shape(W)[0])
            SynapseCount.append(cp.shape(W)[0] * cp.shape(W)[1])
            
        except AttributeError as e:
            raise AttributeError(Fore.RED + "ERROR: W does not have a shape attribute. Check if W is a valid matrix." + Style.RESET_ALL) from e
        except IndexError as e:
            raise IndexError(Fore.RED + "ERROR: W has an unexpected shape format. Ensure it has two dimensions." + Style.RESET_ALL) from e
        except (TypeError, ValueError) as e:
            raise TypeError(Fore.RED + "ERROR: W is not a valid numeric matrix." + Style.RESET_ALL) from e
        except Exception as e:
            raise RuntimeError(Fore.RED + f"ERROR: An unexpected error occurred in save_model: {e}" + Style.RESET_ALL) from e


    elif model_type == 'MLP' or model_type == 'PTNN':

        for i in range(len(W)):
            W[i] = W[i].get()

        W = np.array(W, dtype=object) 

        class_count = W[-1].shape[0]
        
        NeuronCount.append(cp.shape(W[0])[1])

        for i in range(len(W)):
            try:
                NeuronCount.append(cp.shape(W[i])[0])
                SynapseCount.append(cp.shape(W[i])[0] * cp.shape(W[i])[1])

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

        scaler_params[0] = scaler_params[0].get()
        scaler_params[1] = scaler_params[1].get()


    data = {'MODEL NAME': model_name,
            'MODEL TYPE': model_type,
            'CLASS COUNT': class_count,
            'NEURON COUNT': NeuronCount,
            'SYNAPSE COUNT': SynapseCount,
            'TEST ACCURACY': test_acc,
            'VERSION': __version__,
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

                cp.savetxt(model_path + model_name + f'_weights.txt',  W, fmt='%f')

        if weights_type == 'txt' and weights_format == 'raw':

                cp.savetxt(model_path + model_name + f'_weights.txt',  W)

        ###

        
        if weights_type == 'pkl' and weights_format == 'f':

            with open(model_path + model_name + f'_weights.pkl', 'wb') as f:
                pickle.dump(W.astype(float), f)

        if weights_type == 'pkl' and weights_format =='raw':
        
            with open(model_path + model_name + f'_weights.pkl', 'wb') as f:
                pickle.dump(W, f)

        ###

        if weights_type == 'npy' and weights_format == 'f':

                cp.save(model_path + model_name + f'_weights.npy',  W, W.astype(float))

        if weights_type == 'npy' and weights_format == 'raw':

                cp.save(model_path + model_name + f'_weights.npy',  W)

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
    lists: Weights, None, test_accuracy, activations, scaler_params, None, model_type, weight_type, weight_format, device_version, (list[df_elements])=Pandas DataFrame of the model
    """

    from . import __version__
    
    try:

         df = pd.read_pickle(model_path + model_name + '.pkl')

    except OSError as e:
        raise OSError(Fore.RED + f"ERROR: An OSError error occurred in load_model at loading model params. Maybe model name or path or administration issue: {e}" + Style.RESET_ALL) from e


    scaler_params_cpu = df['STANDARD SCALER'].tolist()
    
    try:
        if scaler_params_cpu[0] == None: # model not scaled
            scaler_params = scaler_params_cpu[0]

    except: # model scaled
        scaler_params_cpu = [item for item in scaler_params_cpu if isinstance(item, np.ndarray)]
        scaler_params = cp.array(scaler_params_cpu)

        del scaler_params_cpu
        gc.collect()
     
    model_name = str(df['MODEL NAME'].iloc[0])
    model_type = str(df['MODEL TYPE'].iloc[0])
    WeightType = str(df['WEIGHTS TYPE'].iloc[0])
    WeightFormat = str(df['WEIGHTS FORMAT'].iloc[0])
    test_acc = str(df['TEST ACCURACY'].iloc[0])

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
            Fore.MAGENTA + f"WARNING: Your PyerualJetwork version({device_version}) is different from this model's version({model_version}).\nIf you have a performance issue, please install this model version. Use this: pip install pyerualjetwork=={model_version}" +
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

    if model_type == 'MLP' or model_type == 'PTNN':
        W = W.tolist()
        W = [cp.array(item) for item in W]

    elif model_type == 'PLAN':
        W = cp.array(W)

    return W, None, test_acc, activations, scaler_params, None, model_type, WeightType, WeightFormat, device_version, df, activation_potentiation



def predict_from_storage(Input, model_name, model_path='', dtype=cp.float32):

    """
    Function to make a prediction
    from storage

    Args:
        Input (list or ndarray or cupyarray): Input data for the model.
        
        model_name (str): Name of the model.
        
        model_path (str): Path of the model. Default: ''
        
        dtype (cupy.dtype): Data type for the arrays. cp.float32 by default. Example: cp.float64, cp.float16. (optional)
    Returns:
        cupyarray: Output from the model.
    """
    
    Input = cp.array(Input, dtype=dtype, copy=False)
    
    from .activation_functions import apply_activation
    from .data_ops import standard_scaler
    
    try:

        model = load_model(model_name, model_path)
        
        activations = model[get_act()]
        scaler_params = model[get_scaler()]
        W = model[get_weights()]
        model_type = model[get_model_type()]
        activation_potentiation = model[get_act_pot()]

        if isinstance(activations, str):
            activations = [activations]
        elif isinstance(activations, list):
            activations = [item if isinstance(item, list) or isinstance(item, str) else [item] for item in activations]

        Input = standard_scaler(None, Input, scaler_params)

        if model_type == 'MLP':
            layer = Input
            for i in range(len(W)):
                if i != len(W) - 1 and i != 0: layer = apply_activation(layer, activations[i])

                layer = layer @ W[i].T

            result = layer

        if model_type == 'PLAN':

            Input = apply_activation(Input, activations)
            result = Input @ W.T
        
        if model_type == 'PTNN':

            if isinstance(activation_potentiation, str):
                activation_potentiation = [activation_potentiation]
            elif isinstance(activation_potentiation, list):
                activation_potentiation = [item if isinstance(item, list) or isinstance(item, str) else [item] for item in activation_potentiation]

            Input = apply_activation(Input, activation_potentiation)
            layer = Input @ W[0].T

            for i in range(1, len(W)):
                if i != len(W) - 1: layer = apply_activation(layer, activations[i])

                layer = layer @ W[i].T

            result = layer

        return result

    except Exception as e:
        raise RuntimeError(Fore.RED + f"ERROR: An error occurred in predict_from_storage {e}" + Style.RESET_ALL) from e

    

def reverse_predict_from_storage(output, model_name, model_path='', dtype=cp.float32):

    """
    reverse prediction function from storage
    Args:

        output (list or ndarray): output layer for the model.

        model_name (str): Name of the model.

        model_path (str): Path of the model. Default: ''

        dtype (cupy.dtype): Data type for the arrays. np.float32 by default. Example: cp.float64 or cp.float16. [fp32 for balanced devices, fp64 for strong devices, fp16 for weak devices: not reccomended!] (optional)

    Returns:
        cparray: Input from the model.
    """
    
    output = cp.array(output, dtype=dtype, copy=False)
        
    model = load_model(model_name, model_path)
    
    W = model[get_weights()]

    try:
        Input = W.T @ output
        return Input
    except Exception as e:
        raise RuntimeError(Fore.RED + f"ERROR: An error occurred {e}" + Style.RESET_ALL) from e


def predict_from_memory(Input, W, scaler_params=None, activations=['linear'], activation_potentiation=None, model_type='PLAN'):

    """
    Function to make a prediction
    from memory.

    Args:
        Input (list or ndarray): Input data for the model.
        
        W (list of ndarrays): Weights of the model.
        
        scaler_params (list): standard scaler params list: mean,std. (optional) Default: None.
        
        activations (list[str]): activation list for deep PLAN or activation list for MLP layers. Default: ['linear']
        
        activation_potentiation (list, optional): Extra activation potentiation list (PLAN layers activations) for just PTNN models. Default = None.

        model_type: (str): Type of the model. Options: 'PLAN', 'MLP', 'PTNN'. Default: PLAN

    Returns:
    cupyarray: Output from the model.
    """

    from .data_ops import standard_scaler
    from .activation_functions import apply_activation

    if model_type != 'PLAN' and model_type != 'MLP' and model_type != 'PTNN': raise ValueError("model_type parameter must be 'PLAN', 'MLP' or 'PTNN'.")

    try:

        Input = standard_scaler(None, Input, scaler_params)

        if isinstance(activations, str):
            activations = [activations]
        elif isinstance(activations, list):
            activations = [item if isinstance(item, list) or isinstance(item, str) else [item] for item in activations]

        if model_type == 'MLP':
            layer = Input
            for i in range(len(W)):
                if i != len(W) - 1 and i != 0: layer = apply_activation(layer, activations[i])

                layer = layer @ W[i].T

            result = layer

        if model_type == 'PLAN':

            Input = apply_activation(Input, activations)
            result = Input @ W.T
        
        if model_type == 'PTNN':

            if isinstance(activation_potentiation, str):
                activation_potentiation = [activation_potentiation]
            elif isinstance(activation_potentiation, list):
                activation_potentiation = [item if isinstance(item, list) or isinstance(item, str) else [item] for item in activation_potentiation]

            Input = apply_activation(Input, activation_potentiation)
            layer = Input @ W[0].T

            for i in range(1, len(W)):
                if i != len(W) - 1: layer = apply_activation(layer, activations[i])

                layer = layer @ W[i].T

            result = layer

        return result
    
    except Exception as e:
        raise RuntimeError(Fore.RED + f"ERROR: An error occurred in predict_from_memory {e}" + Style.RESET_ALL) from e
        


def reverse_predict_from_memory(output, W, dtype=cp.float32):

    """
    reverse prediction function from memory

    Args:

        output (list or ndarray): output layer for the model.

        W (list of ndarrays): Weights of the model.

        dtype (cupy.dtype): Data type for the arrays. np.float32 by default. Example: cp.float64 or cp.float16. [fp32 for balanced devices, fp64 for strong devices, fp16 for weak devices: not reccomended!] (optional)

    Returns:
        cparray: Input from the model.
    """

    output = cp.array(output, dtype=dtype, copy=False)

    try:
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