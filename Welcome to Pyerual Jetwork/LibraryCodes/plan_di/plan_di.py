"""
Created on Thu Jun 12 00:00:00 2024

@author: hasan can beydili
"""
import numpy as np
import time
from colorama import Fore,Style
from typing import List, Union
import math
from scipy.special import expit, softmax
import matplotlib.pyplot as plt
import seaborn as sns

# BUILD -----
def fit(
    x_train: List[Union[int, float]],
    y_train: List[Union[int, float, str]], # At least two.. and one hot encoded
) -> str:
        
    infoPLAN = """
    Creates and configures a PLAN model.
    
    Args:
        x_train (list[num]): List of input data.
        y_train (list[num]): List of y_train. (one hot encoded)
        activation_potential (float): Input activation potential 
    
    Returns:
        list([num]): (Weight matrices list, train_predictions list, Train_acc).
        error handled ?: Process status ('e')
"""

    if len(x_train) != len(y_train):
       print(Fore.RED + "ERROR301: x_train list and y_train list must be same length. from: fit",infoPLAN)
       return 'e'
   
    class_count = set()
    for sublist in y_train:
      
        class_count.add(tuple(sublist))
    
    
    class_count = list(class_count)
    
    y_train = [tuple(sublist) for sublist in y_train]
    
    neurons = [len(class_count),len(class_count)]
    layers = ['fex','cat']
    
    x_train[0] = np.array(x_train[0])
    x_train[0] = x_train[0].ravel()
    x_train_size = len(x_train[0])
    
    W = weight_identification(len(layers) - 1,len(class_count),neurons,x_train_size)
    Divides, Piece = synaptic_dividing(len(class_count),W)
    trained_W = [1] * len(W)
    print(Fore.GREEN + "Train Started with 0 ERROR" + Style.RESET_ALL)
    start_time = time.time()
    for index, inp in enumerate(x_train):
        uni_start_time = time.time()
        inp = np.array(inp)
        inp = inp.ravel()
        
        if x_train_size != len(inp):
            print(Fore.RED +"ERROR304: All input matrices or vectors in x_train list, must be same size. from: fit",infoPLAN + Style.RESET_ALL)
            return 'e'
        
        
        for Ulindex, Ul in enumerate(class_count):
            
            if Ul == y_train[index]:
                for Windex, w in enumerate(W):
                    for i, ul in enumerate(Ul):
                        if ul == 1.0:
                            k = i

                    cs = Divides[int(k)][Windex][0]

       
                    W[Windex] = synaptic_pruning(w, cs, 'row', int(k), len(class_count), Piece[Windex], True)

        neural_layer = inp
        
        for Lindex, Layer in enumerate(layers):
            
            
            neural_layer = normalization(neural_layer)
            
            y = np.argmax(y_train[index])
            if Layer == 'fex':
                W[Lindex] = fex(neural_layer, W[Lindex], True, y)
            elif Layer == 'cat':
                W[Lindex] = cat(neural_layer, W[Lindex], True, y)
                
        
        for i, w in enumerate(W):
            trained_W[i] = trained_W[i] + w
         
        W = weight_identification(len(layers) - 1, len(class_count), neurons, x_train_size)
         
               
        uni_end_time = time.time()
        
        calculating_est = round((uni_end_time - uni_start_time) * (len(x_train) - index),3)
        
        if calculating_est < 60:
            print('\rest......(sec):',calculating_est,'\n',end= "")
            
        elif calculating_est > 60 and calculating_est < 3600:
            print('\rest......(min):',calculating_est/60,'\n',end= "")
            
        elif calculating_est > 3600:
            print('\rest......(h):',calculating_est/3600,'\n',end= "")
            
        print('\rTraining: ' , index, "/", len(x_train),"\n", end="")
        
    EndTime = time.time()

    calculating_est = round(EndTime - start_time,2)

    print(Fore.GREEN + " \nTrain Finished with 0 ERROR\n" + Style.RESET_ALL)

    if calculating_est < 60:
     print('Total training time(sec): ',calculating_est)
    
    elif calculating_est > 60 and calculating_est < 3600:
     print('Total training time(min): ',calculating_est/60)
    
    elif calculating_est > 3600:
     print('Total training time(h): ',calculating_est/3600)
    

    return trained_W
        
# FUNCTIONS -----

def weight_identification(
    layer_count,      # int: Number of layers in the neural network.
    class_count,      # int: Number of classes in the classification task.
    neurons,         # list[num]: List of neuron counts for each layer.
    x_train_size        # int: Size of the input data.
) -> str:
    """
    Identifies the weights for a neural network model.

    Args:
        layer_count (int): Number of layers in the neural network.
        class_count (int): Number of classes in the classification task.
        neurons (list[num]): List of neuron counts for each layer.
        x_train_size (int): Size of the input data.

    Returns:
        list([numpy_arrays],[...]): pretrained weight matices of the model. .
    """

    
    Wlen = layer_count + 1
    W = [None] * Wlen
    W[0] = np.ones((neurons[0],x_train_size))
    ws = layer_count - 1
    for w in range(ws):
        W[w + 1] = np.ones((neurons[w + 1],neurons[w]))
    W[layer_count] = np.ones((class_count,neurons[layer_count - 1]))
    return W

def synaptic_pruning(
    w,            # num: Weight matrix of the neural network.
    cs,           # int: cs = cut_start, Synaptic connections between neurons.
    key,          # int: key for identifying synaptic connections.
    Class,        # int: Class label for the current training instance.
    class_count, # int: Total number of classes in the dataset.
    piece, #  int: Which set of neurons will information be transferred to?
    is_training  # bool: Flag indicating if the function is called during training (True or False).
    
) -> str:
    infoPruning = """
    Performs synaptic pruning in a neural network model.

    Args:
        w (list[num]): Weight matrix of the neural network.
        cs (list[num]): Synaptic connections between neurons.
        key (str): key for identifying synaptic row or col connections.
        Class (int): Class label for the current training instance.
        class_count (int): Total number of classes in the dataset.
        piece (int): Which set of neurons will information be transferred to?
        is_training (bool): Flag indicating if the function is called during training (True or False).

    Returns:
        numpy array: Weight matrix.
    """
    
    
    Class += 1 # because index start 0
    
    if  Class != 1:
        
     
               
        ce = cs / Class # ce(cut_end) = cs(cut_start) / current_class
        
        if is_training == True:
        
            p = piece
                
            for i in range(Class - 3):
                    
                piece+=p
                    
            if Class!= 2:
                 ce += piece
            
        w[int(ce)-1::-1,:] = 0
        
            
        w[cs:,:] = 0

    else:
        
            if key == 'row':
    
                w[cs:,:] = 0
    
            elif key == 'col':
    
                w[:,cs] = 0
    
            else:
                print(Fore.RED + "ERROR103: synaptic_pruning func's key parameter must be 'row' or 'col' from: synaptic_pruning" + infoPruning)
                return 'e'
        
    return w

def synaptic_dividing(
    class_count,    # int: Total number of classes in the dataset.
    W              # list[num]: Weight matrix of the neural network.
) -> str:
    """
    Divides the synaptic weights of a neural network model based on class count.

    Args:
        class_count (int): Total number of classes in the dataset.
        W (list[num]): Weight matrix of the neural network.

    Returns:
        list: a 3D list holds informations of divided net and list of neuron groups separated by classes.
    """

    
    Piece = [1] * len(W)
    
    Divides = [[[0] for _ in range(len(W))] for _ in range(class_count)]
    
    
    for i in range(len(W)):
            

            Piece[i] = int(math.floor(W[i].shape[0] / class_count))

    cs = 0 

    for i in range(len(W)):
        for j in range(class_count):
            cs = cs + Piece[i]
            Divides[j][i][0] = cs
            
        j = 0
        cs = 0
        
    return Divides, Piece
        

def fex(
    Input,               # list[num]: Input data.
    w,                   # num: Weight matrix of the neural network.
    is_training,         # bool: Flag indicating if the function is called during training (True or False).
    Class               # int: Which class is, if training. 
) -> tuple:
    """
    Applies feature extraction process to the input data using synaptic pruning.

    Args:
        Input (num): Input data.
        w (num): Weight matrix of the neural network.
        is_training (bool): Flag indicating if the function is called during training (True or False).
        Class (int): if is during training then which class(label) ? is isnt then put None.

    Returns:
        tuple: A tuple (vector) containing the neural layer result and the updated weight matrix.
    """
    
    if is_training == True:
        
        w[Class,:] = Input

        return w
    
    else:
        
        neural_layer = np.dot(w, Input)
        
        return neural_layer

def cat(
    Input,               # list[num]: Input data.
    w,                   # list[num]: Weight matrix of the neural network.
    is_training, # (bool): Flag indicating if the function is called during training (True or False).
    Class
) -> tuple:
    """
    Applies categorization process to the input data using synaptic pruning if specified.

    Args:
        Input (list[num]): Input data.
        w (list[num]): Weight matrix of the neural network.
        is_training (bool): Flag indicating if the function is called during training (True or False).
        Class (int): if is during training then which class(label) ? is isnt then put None.
    Returns:
        tuple: A tuple containing the neural layer (vector) result and the possibly updated weight matrix.
    """
   
    
    
    if is_training == True:
     
        w[Class,Class] += 1
        
        return w
        
    else:
        
        neural_layer = np.dot(w, Input)
    
        return neural_layer


def normalization(
    Input  # num: Input data to be normalized.
):
    """
    Normalizes the input data using maximum absolute scaling.

    Args:
        Input (num): Input data to be normalized.

    Returns:
        (num) Scaled input data after normalization.
    """

   
    AbsVector = np.abs(Input)
    
    MaxAbs = np.max(AbsVector)
    
    ScaledInput = Input / MaxAbs
    
    return ScaledInput


def Softmax(
    x  # num: Input data to be transformed using softmax function.
):
    """
    Applies the softmax function to the input data.

    Args:
        (num): Input data to be transformed using softmax function.

    Returns:
       (num): Transformed data after applying softmax function.
    """
    
    return softmax(x)


def Sigmoid(
    x  # num: Input data to be transformed using sigmoid function.
):
    """
    Applies the sigmoid function to the input data.

    Args:
        (num): Input data to be transformed using sigmoid function.

    Returns:
        (num): Transformed data after applying sigmoid function.
    """
    return expit(x)


def Relu(
    x  # num: Input data to be transformed using ReLU function.
):
    """
    Applies the Rectified Linear Unit (ReLU) function to the input data.

    Args:
        (num): Input data to be transformed using ReLU function.

    Returns:
        (num): Transformed data after applying ReLU function.
    """

    
    return np.maximum(0, x)




def evaluate(
    x_test,         # list[num]: Test input data.
    y_test,         # list[num]: Test labels.
    visualize,         # str: visualize Testing procces or not visualize ('y' or 'n')
    W                  # list[num]: Weight matrix list of the neural network.
) -> tuple:
  infoTestModel =  """
    Tests the neural network model with the given test data.

    Args:
        x_test (list[num]): Test input data.
        y_test (list[num]): Test labels.
        activation_potential (float): Input activation potential 
        visualize (str): Visualize test progress ? ('y' or 'n')
        W (list[num]): Weight matrix list of the neural network.

    Returns:
        tuple: A tuple containing the predicted labels and the accuracy of the model.
    """
    
  layers = ['fex','cat']


  try:
    Wc = [0] * len(W) # Wc = Weight copy
    true = 0
    TestPredictions = [None] * len(y_test)
    for i, w in enumerate(W):
        Wc[i] = np.copy(w)
        print('\rCopying weights.....',i+1,'/',len(W),end = "")
            
    print(Fore.GREEN + "\n\nTest Started with 0 ERROR\n" + Style.RESET_ALL)
    start_time = time.time()
    for inpIndex,Input in enumerate(x_test):
        Input = np.array(Input)
        Input = Input.ravel()
        uni_start_time = time.time()
        neural_layer = Input
        
        for index, Layer in enumerate(layers):
            
            neural_layer = normalization(neural_layer)

            if Layer == 'fex':
                neural_layer = fex(neural_layer, W[index], False, None)
            elif Layer == 'cat':
                neural_layer = cat(neural_layer, W[index], False, None)
                
        for i, w in enumerate(Wc):
            W[i] = np.copy(w)
        RealOutput = np.argmax(y_test[inpIndex])
        PredictedOutput = np.argmax(neural_layer)
        if RealOutput == PredictedOutput:
            true += 1
        acc = true / len(y_test)
        TestPredictions[inpIndex] = PredictedOutput
        
        if visualize == 'y':
        
            y_testVisual = np.copy(y_test) 
            y_testVisual = np.argmax(y_testVisual, axis=1)
            
            plt.figure(figsize=(12, 6))
            sns.kdeplot(y_testVisual, label='Real Outputs', fill=True)
            sns.kdeplot(TestPredictions, label='Predictions', fill=True)
            plt.legend()
            plt.xlabel('Class')
            plt.ylabel('Data size')
            plt.title('Predictions and Real Outputs for Testing KDE Plot')
            plt.show()
            
            if inpIndex + 1 != len(x_test):
            
                plt.close('all')
        
        uni_end_time = time.time()
            
        calculating_est = round((uni_end_time - uni_start_time) * (len(x_test) - inpIndex),3)
            
        if calculating_est < 60:
            print('\rest......(sec):',calculating_est,'\n',end= "")
            print('\rTest accuracy: ' ,acc ,"\n", end="")
        
        elif calculating_est > 60 and calculating_est < 3600:
            print('\rest......(min):',calculating_est/60,'\n',end= "")
            print('\rTest accuracy: ' ,acc ,"\n", end="")
        
        elif calculating_est > 3600:
            print('\rest......(h):',calculating_est/3600,'\n',end= "")
            print('\rTest accuracy: ' ,acc ,"\n", end="")
            
    EndTime = time.time()
    for i, w in enumerate(Wc):
        W[i] = np.copy(w)

    calculating_est = round(EndTime - start_time,2)
    
    print(Fore.GREEN + "\nTest Finished with 0 ERROR\n")
    
    if calculating_est < 60:
        print('Total testing time(sec): ',calculating_est)
        
    elif calculating_est > 60 and calculating_est < 3600:
        print('Total testing time(min): ',calculating_est/60)
        
    elif calculating_est > 3600:
        print('Total testing time(h): ',calculating_est/3600)
        
    if acc >= 0.8:
        print(Fore.GREEN + '\nTotal Test accuracy: ' ,acc, '\n' + Style.RESET_ALL)
    
    elif acc < 0.8 and acc > 0.6:
        print(Fore.MAGENTA + '\nTotal Test accuracy: ' ,acc, '\n' + Style.RESET_ALL)
    
    elif acc <= 0.6:
        print(Fore.RED+ '\nTotal Test accuracy: ' ,acc, '\n' + Style.RESET_ALL)  

        
    
  except:
        
        print(Fore.RED + "ERROR: Are you sure weights are loaded ? from: evaluate" + infoTestModel + Style.RESET_ALL)
        return 'e'
   

   
  return W,TestPredictions,acc

def multiple_evaluate(
    x_test,         # list[num]: Test input data.
    y_test,         # list[num]: Test labels.
    visualize,         # str: visualize Testing procces or not visualize ('y' or 'n')
    MW                  # list[list[num]]: Weight matrix of the neural network.
) -> tuple:
    infoTestModel =  """
    Tests the neural network model with the given test data.

    Args:
        x_test (list[num]): Test input data.
        y_test (list[num]): Test labels.
        activation_potential (float): Input activation potential
        visualize (str): Visualize test progress ? ('y' or 'n')
        MW (list(list[num])): Multiple Weight matrix list of the neural network. (Multiple model testing)

    Returns:
        tuple: A tuple containing the predicted labels and the accuracy of the model.
    """
    
    layers = ['fex','cat']
  
    try:               
        print(Fore.GREEN + "\n\nTest Started with 0 ERROR\n" + Style.RESET_ALL)
        start_time = time.time()
        true = 0
        for inpIndex,Input in enumerate(x_test):
            
            output_layer = 0
            
            for m, Model in enumerate(MW):
                
                W = Model
            
                Wc = [0] * len(W) # Wc = weight copy
                
                TestPredictions = [None] * len(y_test)
                for i, w in enumerate(W):
                    Wc[i] = np.copy(w)
                    
                Input = np.array(Input)
                Input = Input.ravel()
                uni_start_time = time.time()
                neural_layer = Input
                
                for index, Layer in enumerate(layers):
                    
                    neural_layer = normalization(neural_layer)
        
                    if Layer == 'fex':
                        neural_layer = fex(neural_layer, W[index], False, None)
                    elif Layer == 'cat':
                        neural_layer = cat(neural_layer, W[index], False, None)
                    
                output_layer += neural_layer
            
                for i, w in enumerate(Wc):
                    W[i] = np.copy(w)
            for i, w in enumerate(Wc):
                W[i] = np.copy(w)
            RealOutput = np.argmax(y_test[inpIndex])
            PredictedOutput = np.argmax(output_layer)
            if RealOutput == PredictedOutput:
                true += 1
            acc = true / len(y_test)
            TestPredictions[inpIndex] = PredictedOutput
            
            if visualize == 'y':
            
                y_testVisual = np.copy(y_test) 
                y_testVisual = np.argmax(y_testVisual, axis=1)
                
                plt.figure(figsize=(12, 6))
                sns.kdeplot(y_testVisual, label='Real Outputs', fill=True)
                sns.kdeplot(TestPredictions, label='Predictions', fill=True)
                plt.legend()
                plt.xlabel('Class')
                plt.ylabel('Data size')
                plt.title('Predictions and Real Outputs for Testing KDE Plot')
                plt.show()
                
                if inpIndex + 1 != len(x_test):
                
                    plt.close('all')
            
            uni_end_time = time.time()
                
            calculating_est = round((uni_end_time - uni_start_time) * (len(x_test) - inpIndex),3)
                
            if calculating_est < 60:
                print('\rest......(sec):',calculating_est,'\n',end= "")
                print('\rTest accuracy: ' ,acc ,"\n", end="")
            
            elif calculating_est > 60 and calculating_est < 3600:
                print('\rest......(min):',calculating_est/60,'\n',end= "")
                print('\rTest accuracy: ' ,acc ,"\n", end="")
            
            elif calculating_est > 3600:
                print('\rest......(h):',calculating_est/3600,'\n',end= "")
                print('\rTest accuracy: ' ,acc ,"\n", end="")
            
        EndTime = time.time()
        for i, w in enumerate(Wc):
            W[i] = np.copy(w)
    
        calculating_est = round(EndTime - start_time,2)
        
        print(Fore.GREEN + "\nTest Finished with 0 ERROR\n")
        
        if calculating_est < 60:
            print('Total testing time(sec): ',calculating_est)
            
        elif calculating_est > 60 and calculating_est < 3600:
            print('Total testing time(min): ',calculating_est/60)
            
        elif calculating_est > 3600:
            print('Total testing time(h): ',calculating_est/3600)
            
        if acc >= 0.8:
            print(Fore.GREEN + '\nTotal Test accuracy: ' ,acc, '\n' + Style.RESET_ALL)
        
        elif acc < 0.8 and acc > 0.6:
            print(Fore.MAGENTA + '\nTotal Test accuracy: ' ,acc, '\n' + Style.RESET_ALL)
        
        elif acc <= 0.6:
            print(Fore.RED+ '\nTotal Test accuracy: ' ,acc, '\n' + Style.RESET_ALL)  
        
        

    except:
        
            print(Fore.RED + "ERROR: Testing model parameters like 'activation_potential' must be same as trained model. Check parameters. Are you sure weights are loaded ? from: evaluate" + infoTestModel + Style.RESET_ALL)
            return 'e'
   

   
    return W,TestPredictions,acc

def save_model(model_name,
             model_type,
             class_count,
             test_acc,
             weights_type,
             weights_format,
             model_path,
             W
 ):
    
    infosave_model = """
    Function to save a pruning learning model.

    Arguments:
    model_name (str): Name of the model.
    model_type (str): Type of the model.(options: PLAN)
    class_count (int): Number of classes.
    activation_potential (float): Activation potential.
    test_acc (float): Test accuracy of the model.
    weights_type (str): Type of weights to save (options: 'txt', 'npy', 'mat').
    WeightFormat (str): Format of the weights (options: 'd', 'f', 'raw').
    model_path (str): Path where the model will be saved. For example: C:/Users/beydili/Desktop/denemePLAN/
    W: Weights of the model.
    
    Returns:
    str: Message indicating if the model was saved successfully or encountered an error.
    """
    
    # Operations to be performed by the function will be written here
    pass

    layers = ['fex','cat']    

    if weights_type != 'txt' and  weights_type != 'npy' and weights_type != 'mat':
        print(Fore.RED + "ERROR110: Save Weight type (File Extension) Type must be 'txt' or 'npy' or 'mat' from: save_model" + infosave_model + Style.RESET_ALL)
        return 'e'
    
    if weights_format != 'd' and  weights_format != 'f' and weights_format != 'raw':
        print(Fore.RED + "ERROR111: Weight Format Type must be 'd' or 'f' or 'raw' from: save_model" + infosave_model + Style.RESET_ALL)
        return 'e'
    
    NeuronCount = 0
    SynapseCount = 0
    
    try:
        for w in W:
            NeuronCount += np.shape(w)[0]
            SynapseCount += np.shape(w)[0] * np.shape(w)[1]
    except:
        
        print(Fore.RED + "ERROR: Weight matrices has a problem from: save_model" + infosave_model + Style.RESET_ALL)
        return 'e'
    import pandas as pd
    from datetime import datetime
    from scipy import io
    
    data = {'MODEL NAME': model_name,
            'MODEL TYPE': model_type,
            'LAYERS': layers,
            'LAYER COUNT': len(layers),
            'CLASS COUNT': class_count,
            'NEURON COUNT': NeuronCount,
            'SYNAPSE COUNT': SynapseCount,
            'TEST ACCURACY': test_acc,
            'SAVE DATE': datetime.now(),
            'WEIGHTS TYPE': weights_type,
            'WEIGHTS FORMAT': weights_format,
            'MODEL PATH': model_path
            }
    try:
        
        df = pd.DataFrame(data)

            
        df.to_csv(model_path + model_name + '.txt', sep='\t', index=False)
            

    except:
        
        print(Fore.RED + "ERROR: Model log not saved probably model_path incorrect. Check the log parameters from: save_model" + infosave_model + Style.RESET_ALL)
        return 'e'
    try:
        
        if weights_type == 'txt' and weights_format == 'd':
            
            for i, w in enumerate(W):
                np.savetxt(model_path + model_name +  str(i+1) + 'w.txt' ,  w, fmt='%d')
                
        if weights_type == 'txt' and weights_format == 'f':
             
            for i, w in enumerate(W):
                 np.savetxt(model_path + model_name +  str(i+1) + 'w.txt' ,  w, fmt='%f')
        
        if weights_type == 'txt' and weights_format == 'raw':
            
            for i, w in enumerate(W):
                np.savetxt(model_path + model_name +  str(i+1) + 'w.txt' ,  w)
            
                
        ###
        
        
        if weights_type == 'npy' and weights_format == 'd':
            
            for i, w in enumerate(W):
                np.save(model_path + model_name + str(i+1) + 'w.npy', w.astype(int))
        
        if weights_type == 'npy' and weights_format == 'f':
             
            for i, w in enumerate(W):
                 np.save(model_path + model_name +  str(i+1) + 'w.npy' ,  w, w.astype(float))
        
        if weights_type == 'npy' and weights_format == 'raw':
            
            for i, w in enumerate(W):
                np.save(model_path + model_name +  str(i+1) + 'w.npy' ,  w)
                
           
        ###
        
         
        if weights_type == 'mat' and weights_format == 'd':
            
            for i, w in enumerate(W):
                w = {'w': w.astype(int)}
                io.savemat(model_path + model_name + str(i+1) + 'w.mat', w)
    
        if weights_type == 'mat' and weights_format == 'f':
             
            for i, w in enumerate(W):
                w = {'w': w.astype(float)}
                io.savemat(model_path + model_name + str(i+1) + 'w.mat', w)
        
        if weights_type == 'mat' and weights_format == 'raw':
            
            for i, w in enumerate(W):
                w = {'w': w}
                io.savemat(model_path + model_name + str(i+1) + 'w.mat', w)
            
    except:
        
        print(Fore.RED + "ERROR: Model Weights not saved. Check the Weight parameters. SaveFilePath expl: 'C:/Users/hasancanbeydili/Desktop/denemePLAN/' from: save_model" + infosave_model + Style.RESET_ALL)
        return 'e'
    print(df)
    message = (
        Fore.GREEN + "Model Saved Successfully\n" +
        Fore.MAGENTA + "Don't forget, if you want to load model: model log file and weight files must be in the same directory." + 
        Style.RESET_ALL
        )
    
    return print(message)


def load_model(model_name,
             model_path,
):
   infoload_model = """
   Function to load a pruning learning model.

   Arguments:
   model_name (str): Name of the model.
   model_path (str): Path where the model is saved.

   Returns:
   lists: W(list[num]), activation_potential, DataFrame of the model
    """
   pass

    
   import pandas as pd
   import scipy.io as sio
   
   try:

       df = pd.read_csv(model_path + model_name + '.' + 'txt', delimiter='\t')
    
   except:
       
       print(Fore.RED + "ERROR: Model Path error. accaptable form: 'C:/Users/hasancanbeydili/Desktop/denemePLAN/' from: load_model" + infoload_model + Style.RESET_ALL)

   model_name = str(df['MODEL NAME'].iloc[0])
   layers = df['LAYERS'].tolist()
   layer_count = int(df['LAYER COUNT'].iloc[0])
   class_count = int(df['CLASS COUNT'].iloc[0])
   NeuronCount = int(df['NEURON COUNT'].iloc[0])
   SynapseCount = int(df['SYNAPSE COUNT'].iloc[0])
   test_acc = int(df['TEST ACCURACY'].iloc[0])
   model_type = str(df['MODEL TYPE'].iloc[0])
   WeightType = str(df['WEIGHTS TYPE'].iloc[0])
   WeightFormat = str(df['WEIGHTS FORMAT'].iloc[0])
   model_path = str(df['MODEL PATH'].iloc[0])

   W = [0] * layer_count
   
   if WeightType == 'txt':
       for i in range(layer_count):
           W[i] = np.loadtxt(model_path + model_name + str(i+1) + 'w.txt')
   elif WeightType == 'npy':
       for i in range(layer_count):    
           W[i] = np.load(model_path + model_name + str(i+1) + 'w.npy')
   elif WeightType == 'mat':
       for i in range(layer_count):  
           W[i] = sio.loadmat(model_path + model_name + str(i+1) + 'w.mat')
   else:
        raise ValueError(Fore.RED + "Incorrect weight type value. Value must be 'txt', 'npy' or 'mat' from: load_model."  + infoload_model + Style.RESET_ALL)
   print(Fore.GREEN + "Model loaded succesfully" + Style.RESET_ALL)
   return W,df

def predict_model_ssd(Input,model_name,model_path):
    
    infopredict_model_ssd = """
    Function to make a prediction using a divided pruning learning artificial neural network (PLAN).

    Arguments:
    Input (list or ndarray): Input data for the model (single vector or single matrix).
    model_name (str): Name of the model.
    model_path (str): Path where the model is saved.
    Returns:
    ndarray: Output from the model.
    """
    W = load_model(model_name,model_path)[0]
    
    layers = ['fex','cat']
    
    Wc = [0] * len(W)
    for i, w in enumerate(W):
        Wc[i] = np.copy(w)
    try:
        neural_layer = Input
        neural_layer = np.array(neural_layer)
        neural_layer = neural_layer.ravel()
        for index, Layer in enumerate(layers):                                                                          

            neural_layer = normalization(neural_layer)
                                
            if Layer == 'fex':
                neural_layer = fex(neural_layer, W[index], False, None)
            elif Layer == 'cat':
                neural_layer = cat(neural_layer, W[index], False, None)
    except:
       print(Fore.RED + "ERROR: The input was probably entered incorrectly. from: predict_model_ssd"  + infopredict_model_ssd + Style.RESET_ALL)
       return 'e'
    for i, w in enumerate(Wc):
        W[i] = np.copy(w)
    return neural_layer


def predict_model_ram(Input,W):
    
    infopredict_model_ram = """
    Function to make a prediction using a divided pruning learning artificial neural network (PLAN).
    from weights and parameters stored in memory.

    Arguments:
    Input (list or ndarray): Input data for the model (single vector or single matrix).
    activation_potential (float): Activation potential.
    W (list of ndarrays): Weights of the model.

    Returns:
    ndarray: Output from the model.
    """
    
    layers = ['fex','cat']
    
    Wc = [0] * len(W)
    for i, w in enumerate(W):
        Wc[i] = np.copy(w)
    try:
        neural_layer = Input
        neural_layer = np.array(neural_layer)
        neural_layer = neural_layer.ravel()
        for index, Layer in enumerate(layers):                                                                          

            neural_layer = normalization(neural_layer)
                                  
            if Layer == 'fex':
                neural_layer = fex(neural_layer, W[index], False, None)
            elif Layer == 'cat':
                neural_layer = cat(neural_layer, W[index], False, None)
                
    except:
        print(Fore.RED + "ERROR: Unexpected input or wrong model parameters from: predict_model_ram."  + infopredict_model_ram + Style.RESET_ALL)
        return 'e'
    for i, w in enumerate(Wc):
        W[i] = np.copy(w)
    return neural_layer
    

def auto_balancer(x_train, y_train, class_count):
    
   infoauto_balancer = """
   Function to balance the training data across different classes.

   Arguments:
   x_train (list): Input data for training.
   y_train (list): Labels corresponding to the input data.
   class_count (int): Number of classes.

   Returns:
   tuple: A tuple containing balanced input data and labels.
   """
   try:
        ClassIndices = {i: np.where(np.array(y_train)[:, i] == 1)[0] for i in range(class_count)}
        classes = [len(ClassIndices[i]) for i in range(class_count)]
        
        if len(set(classes)) == 1:
            print(Fore.WHITE + "INFO: All training data have already balanced. from: auto_balancer"  + Style.RESET_ALL)
            return x_train, y_train
        
        MinCount = min(classes)
        
        BalancedIndices = []
        for i in range(class_count):
            if len(ClassIndices[i]) > MinCount:
                SelectedIndices = np.random.choice(ClassIndices[i], MinCount, replace=False)
            else:
                SelectedIndices = ClassIndices[i]
            BalancedIndices.extend(SelectedIndices)
        
        BalancedInputs = [x_train[idx] for idx in BalancedIndices]
        BalancedLabels = [y_train[idx] for idx in BalancedIndices]
        
        print(Fore.GREEN + "All Training Data Succesfully Balanced from: " + str(len(x_train)) + " to: " + str(len(BalancedInputs)) + ". from: auto_balancer " + Style.RESET_ALL)
   except:
        print(Fore.RED + "ERROR: Inputs and labels must be same length check parameters" + infoauto_balancer)
        return 'e'
        
   return BalancedInputs, BalancedLabels
   
def synthetic_augmentation(x, y, class_count):
    """
    Generates synthetic examples to balance classes with fewer examples.
    
    Arguments:
    x -- Input dataset (examples) - list format
    y -- Class labels (one-hot encoded) - list format
    class_count -- Number of classes
    
    Returns:
    x_balanced -- Balanced input dataset (list format)
    y_balanced -- Balanced class labels (one-hot encoded, list format)
    """
    # Calculate class distribution
    class_distribution = {i: 0 for i in range(class_count)}
    for label in y:
        class_distribution[np.argmax(label)] += 1
    
    max_class_count = max(class_distribution.values())
    
    x_balanced = list(x)
    y_balanced = list(y)
    
    for class_label in range(class_count):
        class_indices = [i for i, label in enumerate(y) if np.argmax(label) == class_label]
        num_samples = len(class_indices)
        
        if num_samples < max_class_count:
            while num_samples < max_class_count:

                random_indices = np.random.choice(class_indices, 2, replace=False)
                sample1 = x[random_indices[0]]
                sample2 = x[random_indices[1]]
                
                synthetic_sample = sample1 + (np.array(sample2) - np.array(sample1)) * np.random.rand()
                
                x_balanced.append(synthetic_sample.tolist())
                y_balanced.append(y[class_indices[0]])
                
                num_samples += 1
    
    return np.array(x_balanced), np.array(y_balanced)

def standard_scaler(x_train, x_test):
  info_standard_scaler = """
  Standardizes training and test datasets.

  Args:
    train_data: numpy.ndarray
      Training data (n_samples, n_features)
    test_data: numpy.ndarray
      Test data (n_samples, n_features)

  Returns:
    tuple
      Standardized training and test datasets
  """
  try:
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    
    
    train_data_scaled = (x_train - mean) / std
    test_data_scaled = (x_test - mean) / std

  except:
      print(Fore.RED + "ERROR: x_train and x_test must be numpy array from standard_scaler" + info_standard_scaler)
    
  return train_data_scaled, test_data_scaled

   
def get_weights():
        
    return 0
    
def get_df():
        
    return 1
    
def get_preds():
        
    return 1
    
def get_acc():
        
    return 2