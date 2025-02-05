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
               scaler_params=None,
               model_type='PLAN',
               test_acc=None,
               model_path='',
               activation_potentiation=['linear'],
               weights_type='npy',
               weights_format='raw',
               show_architecture=False,
               show_info=True
               ):

    """
    Function to save a potentiation learning artificial neural network model.

    Arguments:

    model_name (str): Name of the model.

    model_type (str): Type of the model. default: 'PLAN'

    test_acc (float): Test accuracy of the model. default: None

    weights_type (str): Type of weights to save (options: 'txt', 'pkl', 'npy', 'mat'). default: 'npy'

    WeightFormat (str): Format of the weights (options: 'f', 'raw'). default: 'raw'

    model_path (str): Path where the model will be saved. For example: C:/Users/beydili/Desktop/denemePLAN/ default: ''

    scaler_params (list[num, num]): standard scaler params list: mean,std. If not used standard scaler then be: None.

    W: Weights of the model.

    activation_potentiation (list): For deeper PLAN networks, activation function parameters. For more information please run this code: plan.activations_list() default: ['linear']

    show_architecture (bool): It draws model architecture. True or False. Default: False

    show_info (bool): Prints model details into console. default: True

    Returns:
    No return.
    """
    
    from .visualizations import draw_model_architecture

    class_count = W.shape[0]

    if test_acc != None:
        test_acc= float(test_acc)

    if weights_type != 'txt' and weights_type != 'npy' and weights_type != 'mat' and weights_type != 'pkl':
        print(Fore.RED + "ERROR110: Save Weight type (File Extension) Type must be 'txt' or 'npy' or 'mat' or 'pkl' from: save_model" + Style.RESET_ALL)
        sys.exit()

    if weights_format != 'd' and weights_format != 'f' and weights_format != 'raw':
        print(Fore.RED + "ERROR111: Weight Format Type must be 'd' or 'f' or 'raw' from: save_model" + Style.RESET_ALL)
        sys.exit()

    NeuronCount = 0
    SynapseCount = 0


    try:
            NeuronCount += np.shape(W)[0] + np.shape(W)[1]
            SynapseCount += np.shape(W)[0] * np.shape(W)[1]
    except:

        print(Fore.RED + "ERROR: Weight matrices has a problem from: save_model" + Style.RESET_ALL)
        sys.exit()

    if scaler_params != None:

        if len(scaler_params) > len(activation_potentiation):

            activation_potentiation += ['']

        elif len(activation_potentiation) > len(scaler_params):

            for i in range(len(activation_potentiation) - len(scaler_params)):

                scaler_params.append(' ')

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
            'ACTIVATION POTENTIATION': activation_potentiation
            }

    df = pd.DataFrame(data)
    df.to_pickle(model_path + model_name + '.pkl')


    try:

        if weights_type == 'txt' and weights_format == 'f':

                np.savetxt(model_path + model_name + '_weights.txt',  W, fmt='%f')

        if weights_type == 'txt' and weights_format == 'raw':

                np.savetxt(model_path + model_name + '_weights.txt',  W)

        ###

        
        if weights_type == 'pkl' and weights_format == 'f':

            with open(model_path + model_name + '_weights.pkl', 'wb') as f:
                pickle.dump(W.astype(float), f)

        if weights_type == 'pkl' and weights_format =='raw':
        
            with open(model_path + model_name + '_weights.pkl', 'wb') as f:
                pickle.dump(W, f)

        ###

        if weights_type == 'npy' and weights_format == 'f':

                np.save(model_path + model_name + '_weights.npy',  W, W.astype(float))

        if weights_type == 'npy' and weights_format == 'raw':

                np.save(model_path + model_name + '_weights.npy',  W)

        ###

        if weights_type == 'mat' and weights_format == 'f':

                w = {'w': W.astype(float)}
                io.savemat(model_path + model_name + '_weights.mat', w)

        if weights_type == 'mat' and weights_format == 'raw':
                
                w = {'w': W}
                io.savemat(model_path + model_name + '_weights.mat', w)

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

   Arguments:

   model_name (str): Name of the model.

   model_path (str): Path where the model is saved.

   Returns:
   lists: W(list[num]), activation_potentiation, DataFrame of the model
    """

    try:

         df = pd.read_pickle(model_path + model_name + '.pkl')

    except:

        print(Fore.RED + "ERROR: Model Path error. acceptable form: 'C:/Users/hasancanbeydili/Desktop/denemePLAN/' from: load_model" + Style.RESET_ALL)

        sys.exit()

    activation_potentiation = list(df['ACTIVATION POTENTIATION'])
    activation_potentiation = [x for x in activation_potentiation if not (isinstance(x, float) and np.isnan(x))]
    activation_potentiation = [item for item in activation_potentiation if item != ''] 

    scaler_params = df['STANDARD SCALER'].tolist()
    
    try:
        if scaler_params[0] == None:
            scaler_params = scaler_params[0]

    except:
        scaler_params = [item for item in scaler_params if isinstance(item, np.ndarray)]

     
    model_name = str(df['MODEL NAME'].iloc[0])
    WeightType = str(df['WEIGHTS TYPE'].iloc[0])

    if WeightType == 'txt':
            W = np.loadtxt(model_path + model_name + '_weights.txt')
    elif WeightType == 'npy':
            W = np.load(model_path + model_name + '_weights.npy')
    elif WeightType == 'mat':
            W = sio.loadmat(model_path + model_name + '_weights.mat')
    elif WeightType == 'pkl':
        with open(model_path + model_name + '_weights.pkl', 'rb') as f:
            W = pickle.load(f)
    else:

        raise ValueError(
            Fore.RED + "Incorrect weight type value. Value must be 'txt', 'npy', 'pkl' or 'mat' from: load_model." + Style.RESET_ALL)
        
    if WeightType == 'mat':
        W = W['w']

    return W, None, None, activation_potentiation, scaler_params


def predict_model_ssd(Input, model_name, model_path='', dtype=np.float32):

    """
    Function to make a prediction using a potentiation learning artificial neural network (PLAN).
    from storage

    Arguments:

    Input (list or ndarray): Input data for the model (single vector or single matrix).

    model_name (str): Name of the model.

    model_path (str): Path of the model. Default: ''

    dtype (numpy.dtype): Data type for the arrays. np.float32 by default. Example: np.float64 or np.float16. [fp32 for balanced devices, fp64 for strong devices, fp16 for weak devices: not reccomended!]

    Returns:
    ndarray: Output from the model.
    """
    
    from .activation_functions import apply_activation
    from .data_operations import standard_scaler
    
    model = load_model(model_name, model_path)
    
    activation_potentiation = model[get_act_pot()]
    scaler_params = model[get_scaler()]
    W = model[get_weights()]

    Input = standard_scaler(None, Input, scaler_params, dtype=dtype)

    Input = np.array(Input, dtype=dtype, copy=False)
    Input = Input.ravel()

    try:
        Input = apply_activation(Input, activation_potentiation)
        neural_layer = Input @ W.T
        return neural_layer
    except:
        print(Fore.RED + "ERROR: Unexpected Output or wrong model parameters from: predict_model_ssd." + Style.RESET_ALL)
        sys.exit()
    

def reverse_predict_model_ssd(output, model_name, model_path='', dtype=np.float32):

    """
    reverse prediction function from storage
    Arguments:

    output (list or ndarray): output layer for the model (single probability vector, output layer of trained model).

    model_name (str): Name of the model.

    model_path (str): Path of the model. Default: ''

    dtype (numpy.dtype): Data type for the arrays. np.float32 by default. Example: np.float64 or np.float16. [fp32 for balanced devices, fp64 for strong devices, fp16 for weak devices: not reccomended!]
    
    Returns:
    ndarray: Input from the model.
    """
    
    model = load_model(model_name, model_path)
    
    W = model[get_weights()]

    try:
        Input = W.T @ output
        return Input
    except:
        print(Fore.RED + "ERROR: Unexpected Output or wrong model parameters from: reverse_predict_model_ssd." + Style.RESET_ALL)
        sys.exit()
    


def predict_model_ram(Input, W, scaler_params=None, activation_potentiation=['linear'], dtype=np.float32):

    """
    Function to make a prediction using a potentiation learning artificial neural network (PLAN).
    from memory.

    Arguments:

    Input (list or ndarray): Input data for the model (single vector or single matrix).

    W (list of ndarrays): Weights of the model.

    scaler_params (list): standard scaler params list: mean,std. (optional) Default: None.
    
    activation_potentiation (list): ac list for deep PLAN. default: [None] ('linear') (optional)

    dtype (numpy.dtype): Data type for the arrays. np.float32 by default. Example: np.float64 or np.float16. [fp32 for balanced devices, fp64 for strong devices, fp16 for weak devices: not reccomended!]

    Returns:
    ndarray: Output from the model.
    """

    from .data_operations import standard_scaler
    from .activation_functions import apply_activation

    Input = standard_scaler(None, Input, scaler_params, dtype=dtype)
    
    Input = np.array(Input, dtype=dtype, copy=False)
    Input = Input.ravel()

    try:
        
        Input = apply_activation(Input, activation_potentiation)
        neural_layer = Input @ W.T

        return neural_layer
    
    except:
        print(Fore.RED + "ERROR: Unexpected input or wrong model parameters from: predict_model_ram." + Style.RESET_ALL)
        sys.exit()

def reverse_predict_model_ram(output, W, dtype=np.float32):

    """
    reverse prediction function from memory

    Arguments:

    output (list or ndarray): output layer for the model (single probability vector, output layer of trained model).

    W (list of ndarrays): Weights of the model.

    dtype (numpy.dtype): Data type for the arrays. np.float32 by default. Example: np.float64 or np.float16. [fp32 for balanced devices, fp64 for strong devices, fp16 for weak devices: not reccomended!]

    Returns:
    ndarray: Input from the model.
    """

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