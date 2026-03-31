"""

Activation Functions on CUDA
============================
This module contains activation functions that run on the CUDA GPU.


Module functions:
-----------------
- 'sigmoid': Sigmoid,
- 'mod_circular': modular_circular_activation,
- 'tanh_circular': tanh_circular_activation,
- 'leaky_relu': leaky_relu,
- 'relu': Relu,
- 'gelu': gelu,
- 'tanh': tanh,
- 'sinakt': sinakt,
- 'p_squared': p_squared,
- 'sglu': lambda x: sglu(x, alpha=1.0),
- 'dlrelu': dlrelu,
- 'sin_plus': sin_plus,
- 'acos': lambda x: acos(x, alpha=1.0, beta=0.0),
- 'isra': isra,
- 'waveakt': waveakt,
- 'arctan': arctan,
- 'bent_identity': bent_identity,
- 'softsign': softsign,
- 'pwl': pwl,
- 'sine': sine,
- 'tanh_square': tanh_square,
- 'linear':,
- 'sine_square': sine_square,
- 'logarithmic': logarithmic,
- 'sine_offset': lambda x: sine_offset(x, 1.0),
- 'spiral': spiral_activation,
- 'circular': circular_activation
- Softmax()
"""

import cupy as cp
import numpy as np
from scipy.special import expit, softmax
import warnings

# ACTIVATION FUNCTIONS ----

def all_activations():
    
    activations_list = ['linear', 'sigmoid', 'relu', 'tanh', 'circular', 'spiral', 'sin_plus', 'mod_circular', 'tanh_circular', 'leaky_relu', 'gelu', 'sinakt', 'p_squared', 'sglu', 'dlrelu', 'acos', 'isra', 'waveakt', 'arctan', 'bent_identity', 'softsign', 'pwl', 'sine', 'tanh_square', 'sine_square', 'logarithmic', 'sine_offset']
    
    return activations_list

def spiral_activation(x):

    r = cp.sqrt(cp.sum(x**2))
    
    theta = cp.arctan2(x[1:], x[:-1])

    spiral_x = r * cp.cos(theta + r)
    spiral_y = r * cp.sin(theta + r)


    spiral_output = cp.concatenate([cp.array([spiral_x[0]]), spiral_y])
    
    return spiral_output


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
    
    return cp.array(softmax(x.get()))


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

    return cp.maximum(0, x)


def tanh(x):
    return cp.tanh(x)

def sin_plus(x):
    return (cp.sin(x) + 1) / 2

def modular_circular_activation(x, period=2*cp.pi):
    return cp.mod(x, period) / period

def gelu(x):
    return 0.5 * x * (1 + cp.tanh(cp.sqrt(2 / cp.pi) * (x + 0.044715 * cp.power(x, 3))))

def tanh_circular_activation(x):
    return (cp.tanh(x) + 1) / 2

def leaky_relu(x, alpha=0.01):
    return cp.where(x > 0, x, alpha * x)

def sinakt(x):
    return cp.sin(x) + cp.cos(x)

def p_squared(x, alpha=1.0, beta=0.0):
    return alpha * x**2 + beta * x

def sglu(x, alpha=1.0):
    return cp.array(softmax(alpha * x.get())) * x

# 4. Double Leaky ReLU (DLReLU)
def dlrelu(x):
    return cp.maximum(0.01 * x, x) + cp.minimum(0.01 * x, 0.1 * x)

# 6. Adaptive Cosine Activation (ACos)
def acos(x, alpha=1.0, beta=0.0):
    return cp.cos(alpha * x + beta)

# 10. Inverse Square Root Activation (ISRA)
def isra(x):
    return x / cp.sqrt(cp.abs(x) + 1)

def waveakt(x, alpha=1.0, beta=2.0, gamma=3.0):
    return cp.sin(alpha * x) * cp.cos(beta * x) * cp.sin(gamma * x)

def arctan(x):
    return cp.arctan(x)

def bent_identity(x):
    return (cp.sqrt(x**2 + 1) - 1) / 2 + x

def circular_activation(x, scale=2.0, frequency=1.0, shift=0.0):    
    
    n_features = x.shape[0]
    
    circular_output = cp.zeros_like(x)
    
    for i in range(n_features):
        
        r = cp.sqrt(cp.sum(x**2))
        theta = 2 * cp.pi * (i / n_features) + shift
        
        circular_x = r * cp.cos(theta + frequency * r) * scale
        circular_y = r * cp.sin(theta + frequency * r) * scale
        
        if i % 2 == 0:
            circular_output[i] = circular_x
        else:
            circular_output[i] = circular_y
    
    return circular_output

def softsign(x):
    return x / (1 + cp.abs(x))

def pwl(x, alpha=0.5, beta=1.5):
    return cp.where(x <= 0, alpha * x, beta * x)

def sine(x, alpha=1.0):
    return cp.sin(alpha * x)

def tanh_square(x):
    return cp.tanh(x)**2

def sine_square(x):
    return cp.sin(x)**2

def logarithmic(x):
    return cp.log(x**2 + 1)

def sine_offset(x, beta=0.0):
    return cp.sin(x + beta)

    
def apply_activation(Input, activation_list):
    """
    Applies activation functions for inputs
    
    Args:
        Input (cupy.ndarray):
        activation_list (list):
    """
    origin_input = cp.copy(Input)
    
    activation_functions = {
        'sigmoid': Sigmoid,
        'mod_circular': modular_circular_activation,
        'tanh_circular': tanh_circular_activation,
        'leaky_relu': leaky_relu,
        'relu': Relu,
        'tanh': tanh,
        'sinakt': sinakt,
        'p_squared': p_squared,
        'sglu': lambda x: sglu(x, alpha=1.0),
        'dlrelu': dlrelu,
        'sin_plus': sin_plus,
        'acos': lambda x: acos(x, alpha=1.0, beta=0.0),
        'isra': isra,
        'waveakt': waveakt,
        'arctan': arctan,
        'bent_identity': bent_identity,
        'softsign': softsign,
        'pwl': pwl,
        'sine': sine,
        'tanh_square': tanh_square,
        'linear': lambda x: x,
        'sine_square': sine_square,
        'logarithmic': logarithmic,
        'sine_offset': lambda x: sine_offset(x, 1.0),
        'spiral': spiral_activation,
        'circular': circular_activation
    }
    
    try:
        
        if isinstance(activation_list, str):
            activation_list = [activation_list]

        activation_list = [str(act).lower() for act in activation_list]

        valid_activations = [act for act in activation_list if act in activation_functions]

        result = origin_input
        for act in valid_activations:
            result = activation_functions[act](result)
            
        return result
                
    except Exception as e:
        warnings.warn(f"Error in activation processing: {str(e)}", RuntimeWarning)
        return Input