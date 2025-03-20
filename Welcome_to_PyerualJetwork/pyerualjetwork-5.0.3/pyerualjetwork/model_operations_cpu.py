""" 


Model Operations on CPU
=======================
This module hosts functions for handling all operational processes related to models on CPU, including:

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
- get_weights_type():
- get_weights_format():
- get_model_version():
- get_model_df():

Examples: https://github.com/HCB06/PyerualJetwork/tree/main/Welcome_to_PyerualJetwork/ExampleCodes

PyerualJetwork document: https://github.com/HCB06/PyerualJetwork/blob/main/Welcome_to_PyerualJetwork/PYERUALJETWORK_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf

- Author: Hasan Can Beydili
- YouTube: https://www.youtube.com/@HasanCanBeydili
- Linkedin: https://www.linkedin.com/in/hasan-can-beydili-77a1b9270/
- Instagram: https://www.instagram.com/canbeydilj
- Contact: tchasancan@gmail.com
"""

import numpy as np
from colorama import Fore, Style
import sys
from datetime import datetime
import pickle
from scipy import io
import scipy.io as sio
import pandas as pd


def save_model(model_name,
               W,
               model_type,
               scaler_params=None,
               test_acc=None,
               model_path='',
               activations=['linear'],
               weights_type='npy',
               weights_format='raw',
               show_architecture=False,
               show_info=True
               ):

    """
    Function to save a potentiation learning artificial neural network model.
    Args:
        model_name: (str): Name of the model.
        
        W: Weights of the model.
        
        model_type: (str): Type of the model. Options: 'PLAN', 'MLP'.
        
        scaler_params: (list[num, num]): standard scaler params list: mean,std. If not used standard scaler then be: None.
        
        test_acc: (float): Test accuracy of the model. default: None
        
        model_path: (str): Path where the model will be saved. For example: C:/Users/beydili/Desktop/denemePLAN/ default: ''
        
        activations: (list[str]): For deeper PLAN networks, activation function parameters. Or activation function parameters for MLP layers. For more information please run this code: neu.activations_list() default: ['linear']
        
        weights_type: (str): Type of weights to save (options: 'txt', 'pkl', 'npy', 'mat'). default: 'npy'
        
        weights_format: (str): Format of the weights (options: 'f', 'raw'). default: 'raw'
        
        show_architecture: (bool): It draws model architecture. True or False. Default: False. NOTE! draw architecture only works for PLAN models. Not MLP models for now, but it will be.
        
        show_info: (bool): Prints model details into console. default: True

    Returns:
        No return.
    """
    
    from .visualizations_cpu import draw_model_architecture

    if model_type != 'PLAN' and model_type != 'MLP':
        raise ValueError("model_type parameter must be 'PLAN' or 'MLP'.")

    if isinstance(activations, str):
        activations = [activations]
    else:
        activations = [item if isinstance(item, list) else [item] for item in activations]

    activation = activations.copy()

    if test_acc != None:
        test_acc= float(test_acc)

    if weights_type != 'txt' and weights_type != 'npy' and weights_type != 'mat' and weights_type != 'pkl':
        print(Fore.RED + "ERROR110: Save Weight type (File Extension) Type must be 'txt' or 'npy' or 'mat' or 'pkl' from: save_model" + Style.RESET_ALL)
        sys.exit()

    if weights_format != 'd' and weights_format != 'f' and weights_format != 'raw':
        print(Fore.RED + "ERROR111: Weight Format Type must be 'd' or 'f' or 'raw' from: save_model" + Style.RESET_ALL)
        sys.exit()

    NeuronCount = []
    SynapseCount = []

    if model_type == 'PLAN':
        class_count = W.shape[0]

        try:
            NeuronCount.append(np.shape(W)[1])
            NeuronCount.append(np.shape(W)[0])
            SynapseCount.append(np.shape(W)[0] * np.shape(W)[1])
        except:

            print(Fore.RED + "ERROR: Weight matrices have a problem from: save_model" + Style.RESET_ALL)
            sys.exit()

    elif model_type == 'MLP':

        class_count = W[-1].shape[0]
        
        NeuronCount.append(np.shape(W[0])[1])

        for i in range(len(W)):
            try:
                    NeuronCount.append(np.shape(W[i])[0])
                    SynapseCount.append(np.shape(W[i])[0] * np.shape(W[i])[1])
            except:

                print(Fore.RED + "ERROR: Weight matrices have a problem from: save_model" + Style.RESET_ALL)
                sys.exit()

    
        SynapseCount.append(' ')
        
        activation.append('')
        activation.insert(0, '')
        
    if len(activation) == 1 and model_type == 'PLAN':
        activation = [activation]
        activation.append('')

    if len(activation) > len(NeuronCount):
        for i in range(len(activation) - len(NeuronCount)):
            NeuronCount.append('')
        
    if len(activation) > len(SynapseCount):
        for i in range(len(activation) - len(SynapseCount)):
            SynapseCount.append('')


    if scaler_params != None:

        if len(scaler_params) > len(activation):

            activation += ['']

        elif len(activation) > len(scaler_params):

            for i in range(len(activation) - len(scaler_params)):

                scaler_params.append(' ')

    from .__init__ import __version__

    data = {'MODEL NAME': model_name,
            'MODEL TYPE': model_type,
            'CLASS COUNT': class_count,
            'NEURON COUNT': NeuronCount,
            'SYNAPSE COUNT': SynapseCount,
            'VERSION': __version__,
            'TEST ACCURACY': test_acc,
            'SAVE DATE': datetime.now(),
            'WEIGHTS TYPE': weights_type,
            'WEIGHTS FORMAT': weights_format,
            'MODEL PATH': model_path,
            'STANDARD SCALER': scaler_params,
            'ACTIVATION FUNCTIONS': activation
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

    except:

        print(Fore.RED + "ERROR: Model Weights not saved. Check the Weight parameters. SaveFilePath expl: 'C:/Users/hasancanbeydili/Desktop/denemePLAN/' from: save_model" + Style.RESET_ALL)
        sys.exit()

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

    from .__init__ import __version__

    try:

         df = pd.read_pickle(model_path + model_name + '.pkl')

    except:

        print(Fore.RED + "ERROR: Model Path or Model Name error. acceptable form: 'C:/Users/hasancanbeydili/Desktop/denemePLAN/' from: load_model" + Style.RESET_ALL)

        sys.exit()

    try:
        activations = list(df['ACTIVATION FUNCTIONS']) # for PyerualJetwork >=5 Versions.
    except KeyError:
        activations = list(df['ACTIVATION POTENTIATION']) # for PyerualJetwork <5 Versions.

    activations = [x for x in activations if not (isinstance(x, float) and np.isnan(x))]
    activations = [item for item in activations if item != ''] 

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
    test_acc = str(df['TEST ACCURACY'].iloc[0])

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

    if model_type == 'MLP': allow_pickle = True
    else: allow_pickle = False

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
        
    if WeightType == 'mat':
        W = W['w']

    return W, None, test_acc, activations, scaler_params, None, model_type, WeightType, WeightFormat, device_version, df



def predict_from_storage(Input, model_name, model_path=''):

    """
    Function to make a prediction
    from storage

    Args:
        Input (list or ndarray): Input data for the model (single vector or single matrix).
        
        model_name (str): Name of the model.
        
        model_path (str): Path of the model. Default: ''
    
    Returns:
        ndarray: Output from the model.
    """
    
    from .activation_functions_cpu import apply_activation
    from .data_operations_cpu import standard_scaler
    
    try:

        model = load_model(model_name, model_path)
        
        activations = model[get_act()]
        scaler_params = model[get_scaler()]
        W = model[get_weights()]
        model_type = model[get_model_type()]

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
            
            return layer

        else:

            Input = apply_activation(Input, activations)
            result = Input @ W.T

            return result

    except:
        print(Fore.RED + "ERROR: Unexpected Output or wrong model parameters from: predict_model_storage." + Style.RESET_ALL)
        sys.exit()


def reverse_predict_from_storage(output, model_name, model_path=''):

    """
    reverse prediction function from storage
    Args:

        output (list or ndarray): output layer for the model (single probability vector, output layer of trained model).

        model_name (str): Name of the model.

        model_path (str): Path of the model. Default: ''

    Returns:
        ndarray: Input from the model.
    """
    
    model = load_model(model_name, model_path)
    
    W = model[get_weights()]

    try:
        Input = W.T @ output
        return Input
    except:
        print(Fore.RED + "ERROR: Unexpected Output or wrong model parameters from: reverse_predict_model_storage." + Style.RESET_ALL)
        sys.exit()
    


def predict_from_memory(Input, W, scaler_params=None, activations=['linear'], is_mlp=False):

    """
    Function to make a prediction.
    from memory.

    Args:
        Input (list or ndarray): Input data for the model (single vector or single matrix).
        
        W (list of ndarrays): Weights of the model.
        
        scaler_params (list): standard scaler params list: mean,std. (optional) Default: None.
        
        activations (list[str]): activation list for deep PLAN or activation list for MLP layers. Default: ['linear']
        
        is_mlp (bool, optional): Predict from PLAN model or MLP model ? Default: False (PLAN)
    Returns:
    ndarray: Output from the model.
    """

    from .data_operations_cpu import standard_scaler
    from .activation_functions_cpu import apply_activation

    try:
    
        Input = standard_scaler(None, Input, scaler_params)
        
        if isinstance(activations, str):
            activations = [activations]
        elif isinstance(activations, list):
            activations = [item if isinstance(item, list) or isinstance(item, str) else [item] for item in activations]
            
        if is_mlp:
            
            layer = Input
            for i in range(len(W)):
                if i != len(W) - 1 and i != 0: layer = apply_activation(layer, activations[i])
                layer = layer @ W[i].T
            
            return layer

        else:

            Input = apply_activation(Input, activations)
            result = Input @ W.T

            return result
        
    except:
        print(Fore.RED + "ERROR: Unexpected input or wrong model parameters from: predict_model_memory." + Style.RESET_ALL)
        sys.exit()

def reverse_predict_from_memory(output, W):

    """
    reverse prediction function from memory

    Args:

        output (list or ndarray): output layer for the model (single probability vector, output layer of trained model).

        W (list of ndarrays): Weights of the model.

    Returns:
        ndarray: Input from the model.
    """

    try:
        Input = W.T @ output
        return Input
    
    except:
        print(Fore.RED + "ERROR: Unexpected Output or wrong model parameters from: reverse_predict_model_memory." + Style.RESET_ALL)
        sys.exit()


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