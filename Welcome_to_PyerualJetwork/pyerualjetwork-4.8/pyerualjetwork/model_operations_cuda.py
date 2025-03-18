import cupy as cp
import numpy as np
from colorama import Fore, Style
import sys
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
               activation_potentiation=['linear'],
               weights_type='npy',
               weights_format='raw',
               show_architecture=None,
               show_info=True
               ):

    """

    Function to save a potentiation learning artificial neural network model or trained non-bias MLP model.
    Args:
        model_name: (str): Name of the model.
        W: Weights of the model.

        model_type: (str): Type of the model. Options: 'PLAN', 'MLP'.
        
        scaler_params: (list[num, num]): standard scaler params list: mean,std. If not used standard scaler then be: None.
        
        model_type: (str): Type of the model. default: 'PLAN'
        
        test_acc: (float): Test accuracy of the model. default: None
        
        model_path: (str): Path where the model will be saved. For example: C:/Users/beydili/Desktop/denemePLAN/ default: ''
       
        activation_potentiation: (list[str]): For deeper PLAN networks, activation function parameters. Or activation function parameters for MLP layers. For more information please run this code: plan.activations_list() default: ['linear']
        
        weights_type: (str): Type of weights to save (options: 'txt', 'pkl', 'npy', 'mat'). default: 'npy'
        
        weights_format: (str): Format of the weights (options: 'f', 'raw'). default: 'raw'
        
        show_architecture: (bool): It draws model architecture. True or False. Default: False. NOTE! draw architecture only works for PLAN models. Not MLP models for now, but it will be.
        
        show_info: (bool): Prints model details into console. default: True


    Returns:
        No return.
    """
    
    from .visualizations_cuda import draw_model_architecture

    if model_type != 'PLAN' and model_type != 'MLP':
        raise ValueError("model_type parameter must be 'PLAN' or 'MLP'.")

    if isinstance(W, list):
        W = W.copy()

    if isinstance(W, np.ndarray):
        W = np.copy(W)
        
    if isinstance(activation_potentiation, str):
        activation_potentiation = [activation_potentiation]
    else:
        activation_potentiation = [item if isinstance(item, list) else [item] for item in activation_potentiation]

    activation = activation_potentiation.copy()

    if test_acc is not None:
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
            NeuronCount.append(cp.shape(W)[1])
            NeuronCount.append(cp.shape(W)[0])
            SynapseCount.append(cp.shape(W)[0] * cp.shape(W)[1])
        except:

            print(Fore.RED + "ERROR: Weight matrices have a problem from: save_model" + Style.RESET_ALL)
            sys.exit()

    elif model_type == 'MLP':

        for i in range(len(W)):
            W[i] = W[i].get()

        W = np.array(W, dtype=object) 

        class_count = W[-1].shape[0]
        
        NeuronCount.append(cp.shape(W[0])[1])

        for i in range(len(W)):
            try:
                NeuronCount.append(cp.shape(W[i])[0])
                SynapseCount.append(cp.shape(W[i])[0] * cp.shape(W[i])[1])
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

        scaler_params[0] = scaler_params[0].get()
        scaler_params[1] = scaler_params[1].get()

    data = {'MODEL NAME': model_name,
            'MODEL TYPE': model_type,
            'CLASS COUNT': class_count,
            'NEURON COUNT': NeuronCount,
            'SYNAPSE COUNT': SynapseCount,
            'TEST ACCURACY': test_acc,
            'SAVE DATE': datetime.now(),
            'WEIGHTS TYPE': weights_type,
            'WEIGHTS FORMAT': weights_format,
            'MODEL PATH': model_path,
            'STANDARD SCALER': scaler_params,
            'ACTIVATION POTENTIATION': activation
            }

    df = pd.DataFrame(data)
    df.to_pickle(model_path + model_name + '.pkl')


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
    lists: (list[df_elements]), DataFrame of the model
    """
    
    try:

         df = pd.read_pickle(model_path + model_name + '.pkl')

    except:

        print(Fore.RED + "ERROR: Model Path error. acceptable form: 'C:/Users/hasancanbeydili/Desktop/denemePLAN/' from: load_model" + Style.RESET_ALL)

        sys.exit()

    activation_potentiation = list(df['ACTIVATION POTENTIATION'])
    activation_potentiation = [x for x in activation_potentiation if not (isinstance(x, float) and cp.isnan(x))]
    activation_potentiation = [item for item in activation_potentiation if item != '']

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

    if model_type == 'MLP':
        W = W.tolist()
        W = [cp.array(item) for item in W]

    return W, None, None, activation_potentiation, scaler_params, None, model_type



def predict_model_ssd(Input, model_name, model_path='', dtype=cp.float32):

    """
    Function to make a prediction using a potentiation learning artificial neural network (PLAN).
    from storage

    Args:
        Input (list or ndarray or cupyarray): Input data for the model (single vector or single matrix).
        
        model_name (str): Name of the model.
        
        model_path (str): Path of the model. Default: ''
        
        dtype (cupy.dtype): Data type for the arrays. cp.float32 by default. Example: cp.float64, cp.float16. (optional)
    Returns:
        cupyarray: Output from the model.
    """
    
    Input = cp.array(Input, dtype=dtype, copy=False)
    
    from .activation_functions_cuda import apply_activation
    from .data_operations_cuda import standard_scaler
    
    try:

        model = load_model(model_name, model_path)
        
        activation_potentiation = model[get_act_pot()]
        scaler_params = model[get_scaler()]
        W = model[get_weights()]
        model_type = model[get_model_type()]

        if isinstance(activation_potentiation, str):
            activation_potentiation = [activation_potentiation]
        elif isinstance(activation_potentiation, list):
            activation_potentiation = [item if isinstance(item, list) or isinstance(item, str) else [item] for item in activation_potentiation]

        Input = standard_scaler(None, Input, scaler_params)

        if model_type == 'MLP':
            
            layer = Input
            for i in range(len(W)):
                if i != len(W) - 1 and i != 0: layer = apply_activation(layer, activation_potentiation[i])
                layer = layer @ W[i].T
            
            return layer

        else:

            Input = apply_activation(Input, activation_potentiation)
            result = Input @ W.T

            return result

    except:
        print(Fore.RED + "ERROR: Unexpected Output or wrong model parameters from: predict_model_ssd." + Style.RESET_ALL)
        sys.exit(),

    

def reverse_predict_model_ssd(output, model_name, model_path='', dtype=cp.float32):

    """
    reverse prediction function from storage
    Args:

        output (list or ndarray): output layer for the model (single probability vector, output layer of trained model).

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
    except:
        print(Fore.RED + "ERROR: Unexpected Output or wrong model parameters from: reverse_predict_model_ssd." + Style.RESET_ALL)
        sys.exit()


def predict_model_ram(Input, W, scaler_params=None, activation_potentiation=['linear'], is_mlp=False):

    """
    Function to make a prediction using a potentiation learning artificial neural network (PLAN).
    from memory.

    Args:
        Input (list or ndarray): Input data for the model (single vector or single matrix).
        
        W (list of ndarrays): Weights of the model.
        
        scaler_params (list): standard scaler params list: mean,std. (optional) Default: None.
        
        activation_potentiation (list[str]): activation list for deep PLAN or activation list for MLP layers. Default: ['linear']
        
        is_mlp (bool, optional): Predict from PLAN model or MLP model ? Default: False (PLAN)
    Returns:
    cupyarray: Output from the model.
    """

    from .data_operations_cuda import standard_scaler
    from .activation_functions_cuda import apply_activation

    try:
        
        if isinstance(activation_potentiation, str):
            activation_potentiation = [activation_potentiation]
        elif isinstance(activation_potentiation, list):
            activation_potentiation = [item if isinstance(item, list) or isinstance(item, str) else [item] for item in activation_potentiation]
            
        Input = standard_scaler(None, Input, scaler_params)
        
        if is_mlp:
            
            layer = Input
            for i in range(len(W)):
                if i != len(W) - 1 and i != 0: layer = apply_activation(layer, activation_potentiation[i])
                layer = layer @ W[i].T
            
            return layer

        else:

            Input = apply_activation(Input, activation_potentiation)
            result = Input @ W.T

            return result
        
    except:
        print(Fore.RED + "ERROR: Unexpected input or wrong model parameters from: predict_model_ram." + Style.RESET_ALL)
        sys.exit()


def reverse_predict_model_ram(output, W, dtype=cp.float32):

    """
    reverse prediction function from memory

    Args:

        output (list or ndarray): output layer for the model (single probability vector, output layer of trained model).

        W (list of ndarrays): Weights of the model.

        dtype (cupy.dtype): Data type for the arrays. np.float32 by default. Example: cp.float64 or cp.float16. [fp32 for balanced devices, fp64 for strong devices, fp16 for weak devices: not reccomended!] (optional)

    Returns:
        cparray: Input from the model.
    """

    output = cp.array(output, dtype=dtype, copy=False)

    try:
        Input = W.T @ output
        return Input
    
    except:
        print(Fore.RED + "ERROR: Unexpected Output or wrong model parameters from: reverse_predict_model_ram." + Style.RESET_ALL)
        sys.exit()


def get_weights():

    return 0


def get_preds():

    return 1


def get_acc():

    return 2


def get_act_pot():

    return 3


def get_scaler():

    return 4

def get_preds_softmax():

    return 5

def get_model_type():
    return 6