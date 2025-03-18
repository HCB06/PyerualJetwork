import cupy as cp
from scipy.special import expit, softmax
import warnings

# ACTIVATION FUNCTIONS ----

def all_activations():
    
    activations_list = ['linear', 'sigmoid', 'relu', 'tanh', 'circular', 'spiral', 'swish', 'sin_plus', 'mod_circular', 'tanh_circular', 'leaky_relu', 'softplus', 'elu', 'gelu', 'selu', 'sinakt', 'p_squared', 'sglu', 'dlrelu', 'exsig', 'acos', 'srelu', 'qelu', 'isra', 'waveakt', 'arctan', 'bent_identity', 'sech', 'softsign', 'pwl', 'cubic', 'gaussian', 'sine', 'tanh_square', 'mod_sigmoid', 'quartic', 'square_quartic', 'cubic_quadratic', 'sine_square', 'logarithmic', 'scaled_cubic', 'sine_offset']
    
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

def swish(x):
    return x * (1 / (1 + cp.exp(-x)))

def sin_plus(x):
    return (cp.sin(x) + 1) / 2

def modular_circular_activation(x, period=2*cp.pi):
    return cp.mod(x, period) / period

def tanh_circular_activation(x):
    return (cp.tanh(x) + 1) / 2

def leaky_relu(x, alpha=0.01):
    return cp.where(x > 0, x, alpha * x)

def softplus(x):
    return cp.log(1 + cp.exp(x))

def elu(x, alpha=1.0):
    return cp.where(x > 0, x, alpha * (cp.exp(x) - 1))

def gelu(x):
    return 0.5 * x * (1 + cp.tanh(cp.sqrt(2 / cp.pi) * (x + 0.044715 * cp.power(x, 3))))

def selu(x, lambda_=1.0507, alpha=1.6733):
    return lambda_ * cp.where(x > 0, x, alpha * (cp.exp(x) - 1))

def sinakt(x):
    return cp.sin(x) + cp.cos(x)

def p_squared(x, alpha=1.0, beta=0.0):
    return alpha * x**2 + beta * x

def sglu(x, alpha=1.0):
    return cp.array(softmax(alpha * x.get())) * x

# 4. Double Leaky ReLU (DLReLU)
def dlrelu(x):
    return cp.maximum(0.01 * x, x) + cp.minimum(0.01 * x, 0.1 * x)

# 5. Exponential Sigmoid (ExSig)
def exsig(x):
    return 1 / (1 + cp.exp(-x**2))

# 6. Adaptive Cosine Activation (ACos)
def acos(x, alpha=1.0, beta=0.0):
    return cp.cos(alpha * x + beta)

# 7. Gaussian-like Activation (GLA)
def gla(x, alpha=1.0, mu=0.0):
    return cp.exp(-alpha * (x - mu)**2)

# 8. Swish ReLU (SReLU)
def srelu(x):
    return x * (1 / (1 + cp.exp(-x))) + cp.maximum(0, x)

# 9. Quadratic Exponential Linear Unit (QELU)
def qelu(x):
    return x**2 * cp.exp(x) - 1

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

def sech(x):
    return 2 / (cp.exp(x) + cp.exp(-x))

def softsign(x):
    return x / (1 + cp.abs(x))

def pwl(x, alpha=0.5, beta=1.5):
    return cp.where(x <= 0, alpha * x, beta * x)

def cubic(x):
    return x**3

def gaussian(x, alpha=1.0, mu=0.0):
    return cp.exp(-alpha * (x - mu)**2)
                
def sine(x, alpha=1.0):
    return cp.sin(alpha * x)

def tanh_square(x):
    return cp.tanh(x)**2

def mod_sigmoid(x, alpha=1.0, beta=0.0):
    return 1 / (1 + cp.exp(-alpha * x + beta))

def quartic(x):
    return x**4

def square_quartic(x):
    return (x**2)**2

def cubic_quadratic(x):
    return x**3 * (x**2)

def sine_square(x):
    return cp.sin(x)**2

def logarithmic(x):
    return cp.log(x**2 + 1)

def scaled_cubic(x, alpha=1.0):
    return alpha * x**3

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
        'swish': swish,
        'mod_circular': modular_circular_activation,
        'tanh_circular': tanh_circular_activation,
        'leaky_relu': leaky_relu,
        'relu': Relu,
        'softplus': softplus,
        'elu': elu,
        'gelu': gelu,
        'selu': selu,
        'tanh': tanh,
        'sinakt': sinakt,
        'p_squared': p_squared,
        'sglu': lambda x: sglu(x, alpha=1.0),
        'dlrelu': dlrelu,
        'exsig': exsig,
        'sin_plus': sin_plus,
        'acos': lambda x: acos(x, alpha=1.0, beta=0.0),
        'gla': lambda x: gla(x, alpha=1.0, mu=0.0),
        'srelu': srelu,
        'qelu': qelu,
        'isra': isra,
        'waveakt': waveakt,
        'arctan': arctan,
        'bent_identity': bent_identity,
        'sech': sech,
        'softsign': softsign,
        'pwl': pwl,
        'cubic': cubic,
        'gaussian': gaussian,
        'sine': sine,
        'tanh_square': tanh_square,
        'mod_sigmoid': mod_sigmoid,
        'linear': lambda x: x,
        'quartic': quartic,
        'square_quartic': square_quartic,
        'cubic_quadratic': cubic_quadratic,
        'sine_square': sine_square,
        'logarithmic': logarithmic,
        'scaled_cubic': lambda x: scaled_cubic(x, 1.0),
        'sine_offset': lambda x: sine_offset(x, 1.0),
        'spiral': spiral_activation,
        'circular': circular_activation
    }
    
    try:
        
        valid_mask = cp.array([act in activation_functions for act in activation_list])
        valid_activations = cp.array(activation_list)[valid_mask]

        activation_outputs = cp.array([activation_functions[act](origin_input) for act in valid_activations])

        return cp.sum(activation_outputs, axis=0)
        
    except Exception as e:
        warnings.warn(f"Error in activation processing: {str(e)}", RuntimeWarning)
        return Input
