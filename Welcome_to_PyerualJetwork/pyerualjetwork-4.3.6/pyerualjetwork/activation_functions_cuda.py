import cupy as cp
from scipy.special import expit, softmax
import warnings

# ACTIVATION FUNCTIONS ----

def all_activations():
    
    activations_list = ['linear', 'sigmoid', 'relu', 'tanh', 'circular', 'spiral', 'swish', 'sin_plus', 'mod_circular', 'tanh_circular', 'leaky_relu', 'softplus', 'elu', 'gelu', 'selu', 'sinakt', 'p_squared', 'sglu', 'dlrelu', 'exsig', 'acos', 'gla', 'srelu', 'qelu', 'isra', 'waveakt', 'arctan', 'bent_identity', 'sech', 'softsign', 'pwl', 'cubic', 'gaussian', 'sine', 'tanh_square', 'mod_sigmoid', 'quartic', 'square_quartic', 'cubic_quadratic', 'exp_cubic', 'sine_square', 'logarithmic', 'scaled_cubic', 'sine_offset']
    
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

def exp_cubic(x):
    return cp.exp(x**3)

def sine_square(x):
    return cp.sin(x)**2

def logarithmic(x):
    return cp.log(x**2 + 1)

def scaled_cubic(x, alpha=1.0):
    return alpha * x**3

def sine_offset(x, beta=0.0):
    return cp.sin(x + beta)


def safe_add(current_sum, new_value):
        try:
            return current_sum + new_value
        except OverflowError:
            return cp.array(current_sum) + cp.array(new_value)
    
def apply_activation(Input, activation_list):
    """
        Applies a sequence of activation functions to the input.
        
        Args:
            Input (cupy.ndarray): The input to apply activations to.
            activation_list (list): A list of activation function names to apply.
        
        Returns:
            cupy.ndarray: The input after all activations have been applied.
    """
    
    origin_input = cp.copy(Input)

    for i in range(len(activation_list)):
        try:
            if activation_list[i] == 'sigmoid':
                Input = safe_add(Input, Sigmoid(origin_input))
            elif activation_list[i] == 'swish':
                Input = safe_add(Input, swish(origin_input))
            elif activation_list[i] == 'mod_circular':
                Input = safe_add(Input, modular_circular_activation(origin_input))
            elif activation_list[i] == 'tanh_circular':
                Input = safe_add(Input, tanh_circular_activation(origin_input))
            elif activation_list[i] == 'leaky_relu':
                Input = safe_add(Input, leaky_relu(origin_input))
            elif activation_list[i] == 'relu':
                Input = safe_add(Input, Relu(origin_input))
            elif activation_list[i] == 'softplus':
                Input = safe_add(Input, softplus(origin_input))
            elif activation_list[i] == 'elu':
                Input = safe_add(Input, elu(origin_input))
            elif activation_list[i] == 'gelu':
                Input = safe_add(Input, gelu(origin_input))
            elif activation_list[i] == 'selu':
                Input = safe_add(Input, selu(origin_input))
            elif activation_list[i] == 'tanh':
                Input = safe_add(Input, tanh(origin_input))
            elif activation_list[i] == 'sinakt':
                Input = safe_add(Input, sinakt(origin_input))
            elif activation_list[i] == 'p_squared':
                Input = safe_add(Input, p_squared(origin_input))
            elif activation_list[i] == 'sglu':
                Input = safe_add(Input, sglu(origin_input, alpha=1.0))
            elif activation_list[i] == 'dlrelu':
                Input = safe_add(Input, dlrelu(origin_input))
            elif activation_list[i] == 'exsig':
                Input = safe_add(Input, exsig(origin_input))
            elif activation_list[i] == 'sin_plus':
                Input = safe_add(Input, sin_plus(origin_input))
            elif activation_list[i] == 'acos':
                Input = safe_add(Input, acos(origin_input, alpha=1.0, beta=0.0))
            elif activation_list[i] == 'gla':
                Input = safe_add(Input, gla(origin_input, alpha=1.0, mu=0.0))
            elif activation_list[i] == 'srelu':
                Input = safe_add(Input, srelu(origin_input))
            elif activation_list[i] == 'qelu':
                Input = safe_add(Input, qelu(origin_input))
            elif activation_list[i] == 'isra':
                Input = safe_add(Input, isra(origin_input))
            elif activation_list[i] == 'waveakt':
                Input = safe_add(Input, waveakt(origin_input))
            elif activation_list[i] == 'arctan':
                Input = safe_add(Input, arctan(origin_input))
            elif activation_list[i] == 'bent_identity':
                Input = safe_add(Input, bent_identity(origin_input))
            elif activation_list[i] == 'sech':
                Input = safe_add(Input, sech(origin_input))
            elif activation_list[i] == 'softsign':
                Input = safe_add(Input, softsign(origin_input))
            elif activation_list[i] == 'pwl':
                Input = safe_add(Input, pwl(origin_input))
            elif activation_list[i] == 'cubic':
                Input = safe_add(Input, cubic(origin_input))
            elif activation_list[i] == 'gaussian':
                Input = safe_add(Input, gaussian(origin_input))
            elif activation_list[i] == 'sine':
                Input = safe_add(Input, sine(origin_input))
            elif activation_list[i] == 'tanh_square':
                Input = safe_add(Input, tanh_square(origin_input))
            elif activation_list[i] == 'mod_sigmoid':
                Input = safe_add(Input, mod_sigmoid(origin_input))
            elif activation_list[i] == 'linear':
                Input = safe_add(Input, origin_input)
            elif activation_list[i] == 'quartic':
                Input = safe_add(Input, quartic(origin_input))
            elif activation_list[i] == 'square_quartic':
                Input = safe_add(Input, square_quartic(origin_input))
            elif activation_list[i] == 'cubic_quadratic':
                Input = safe_add(Input, cubic_quadratic(origin_input))
            elif activation_list[i] == 'exp_cubic':
                Input = safe_add(Input, exp_cubic(origin_input))
            elif activation_list[i] == 'sine_square':
                Input = safe_add(Input, sine_square(origin_input))
            elif activation_list[i] == 'logarithmic':
                Input = safe_add(Input, logarithmic(origin_input))
            elif activation_list[i] == 'scaled_cubic':
                Input = safe_add(Input, scaled_cubic(origin_input, 1.0))
            elif activation_list[i] == 'sine_offset':
                Input = safe_add(Input, sine_offset(origin_input, 1.0))
            elif activation_list[i] == 'spiral':
                Input = safe_add(Input, spiral_activation(origin_input))
            elif activation_list[i] == 'circular':
                Input = safe_add(Input, circular_activation(origin_input))
        
        except Exception as e:
            warnings.warn(f"Error in activation {activation_list[i]}: {str(e)}", RuntimeWarning)
            if not isinstance(Input, cp.ndarray):
                Input = cp.array(Input)
            if not isinstance(origin_input, cp.ndarray):
                origin_input = cp.array(origin_input)
            continue
 
    return Input