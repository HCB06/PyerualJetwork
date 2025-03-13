from tqdm import tqdm
import cupy as cp
from colorama import Fore, Style
import math
import numpy as np

def encode_one_hot(y_train, y_test=None, summary=False):
    """
    Performs one-hot encoding on y_train and y_test data.

    Args:
        y_train (cupy.ndarray): Train label data.
        y_test (cupy.ndarray): Test label data. (optional).
        summary (bool): If True, prints the class-to-index mapping. Default: False
   
    Returns:
        tuple: One-hot encoded y_train and (if given: y_test).
    """

    from .memory_operations import optimize_labels, transfer_to_cpu
    
    y_train = transfer_to_cpu(y_train,dtype=y_train.dtype)
    y_test = transfer_to_cpu(y_test,dtype=y_test.dtype)

    classes = np.unique(y_train)
    class_count = len(classes)

    class_to_index = {cls: idx for idx, cls in enumerate(classes)}

    if summary:
        print("Class-to-index mapping:")
        for cls, idx in class_to_index.items():
            print(f"  {idx}: {cls}")

    y_train_encoded = np.zeros((y_train.shape[0], class_count), dtype=y_train.dtype)
    for i, label in enumerate(y_train):
        y_train_encoded[i, class_to_index[label]] = 1
    y_train_encoded = optimize_labels(y_train_encoded, one_hot_encoded=True, cuda=True)

    if y_test is not None:
        y_test_encoded = np.zeros((y_test.shape[0], class_count), dtype=y_test.dtype)
        for i, label in enumerate(y_test):
            y_test_encoded[i, class_to_index[label]] = 1
        y_test_encoded = optimize_labels(y_test_encoded, one_hot_encoded=True, cuda=True)

        return y_train_encoded, y_test_encoded

    return y_train_encoded


def decode_one_hot(encoded_data):
    """
    Decodes one-hot encoded data to original categorical labels.

    Args:
        encoded_data (cupy.ndarray): One-hot encoded data with shape (n_samples, n_classes).

    Returns:
        cupy.ndarray: Decoded categorical labels with shape (n_samples,).
    """

    if encoded_data.ndim == 1: return cp.argmax(encoded_data)
    else: return cp.argmax(encoded_data, axis=1)



def split(X, y, test_size, random_state=42, dtype=cp.float32, shuffle_in_cpu=False):
    """
    Splits the given X (features) and y (labels) data into training and testing subsets.

    Args:
        X (cupy.ndarray): Features data.
        
        y (cupy.ndarray): Labels data.
        
        test_size (float or int): Proportion or number of samples for the test subset.
        
        random_state (int or None): Seed for random state. Default: 42.
        
        dtype (cupy.dtype): Data type for the arrays. np.float32 by default. Example: cp.float64 or cp.float16.
        
        shuffle_in_cpu (bool): If True, output will be same cpu's split function. Default: False. (Use this for direct comparison of cpu training.)
    Returns:
        tuple: x_train, x_test, y_train, y_test as ordered training and testing data subsets.
    """
    from .memory_operations import transfer_to_gpu, optimize_labels
    
    X = transfer_to_gpu(X, dtype=dtype)
    y = optimize_labels(y, one_hot_encoded=False, cuda=True)
    
    num_samples = X.shape[0]

    if isinstance(test_size, float):
        test_size = int(test_size * num_samples)
    elif isinstance(test_size, int):
        if test_size > num_samples:
            raise ValueError(
                "test_size cannot be larger than the number of samples.")
    else:
        raise ValueError("test_size should be float or int.")

    if shuffle_in_cpu:
        indices = np.arange(num_samples)
        np.random.seed(random_state)
        np.random.shuffle(indices)
        indices = cp.array(indices)
    else:
        if random_state is not None:
            cp.random.seed(random_state)

        indices = cp.arange(num_samples)
        cp.random.shuffle(indices)

    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    x_train, x_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    del X
    del y
    cp.cuda.MemoryPool().free_all_blocks()
    return x_train, x_test, y_train, y_test


def manuel_balancer(x_train, y_train, target_samples_per_class, dtype=cp.float32, shuffle_in_cpu=False):
    """
    Generates synthetic examples to balance classes to the specified number of examples per class.

    Args:

        x_train: cupy array format

        y_train (one-hot encoded): cupy array format

        target_samples_per_class (int): Desired number of samples per class

        dtype (cupy.dtype): Data type for the arrays. cp.float32 by default. Example: cp.float64 or cp.float16.

        shuffle_in_cpu (bool): If True, output will be same cpu's manuel_balancer function. Default: False. (Use this for direct comparison of cpu training.)

    Returns:
        x_balanced -- Balanced input dataset (cupy array format)
        y_balanced -- Balanced class labels (one-hot encoded, cupy array format)
    """
    from .ui import loading_bars, get_loading_bar_style
    from .memory_operations import transfer_to_gpu
    
    bar_format = loading_bars()[0]
    x_train = transfer_to_gpu(x_train, dtype=dtype)
    y_train = transfer_to_gpu(y_train, dtype=y_train.dtype)

    classes = cp.arange(y_train.shape[1])
    class_count = len(classes)
    
    x_balanced = []
    y_balanced = []

    for class_label in tqdm(range(class_count),leave=False, ascii=get_loading_bar_style(),
            bar_format=bar_format,desc='Augmenting Data',ncols= 52):
        class_indices = cp.where(cp.argmax(y_train, axis=1) == class_label)[0]
        num_samples = len(class_indices)
        
        if num_samples > target_samples_per_class:
            
            if shuffle_in_cpu:
                selected_indices = np.random.choice(
                    class_indices.get(), target_samples_per_class, replace=False)
            else:
                selected_indices = cp.random.choice(class_indices, target_samples_per_class, replace=False)

            x_balanced.append(x_train[selected_indices])
            y_balanced.append(y_train[selected_indices])
            
        else:
            
            x_balanced.append(x_train[class_indices])
            y_balanced.append(y_train[class_indices])

            if num_samples < target_samples_per_class:
                
                samples_to_add = target_samples_per_class - num_samples
                additional_samples = cp.zeros((samples_to_add, x_train.shape[1]), dtype=x_train.dtype)
                additional_labels = cp.zeros((samples_to_add, y_train.shape[1]), dtype=y_train.dtype)
                
                for i in range(samples_to_add):
                    
                    if shuffle_in_cpu:
                        random_indices = np.random.choice(class_indices.get(), 2, replace=False)
                    else:
                        random_indices = cp.random.choice(class_indices, 2, replace=False)

                    sample1 = x_train[random_indices[0]]
                    sample2 = x_train[random_indices[1]]

                    if shuffle_in_cpu:
                        synthetic_sample = sample1 + (sample2 - sample1) * np.random.rand()
                    else:
                        synthetic_sample = sample1 + (sample2 - sample1) * cp.random.rand()

                    additional_samples[i] = synthetic_sample
                    additional_labels[i] = y_train[class_indices[0]]
                    
                    
                x_balanced.append(additional_samples)
                y_balanced.append(additional_labels)
    
    x_balanced = cp.vstack(x_balanced, dtype=x_train.dtype)
    y_balanced = cp.vstack(y_balanced, dtype=y_train.dtype)

    del x_train, y_train
    cp.cuda.MemoryPool().free_all_blocks()

    return x_balanced, y_balanced


def auto_balancer(x_train, y_train, dtype=cp.float32, shuffle_in_cpu=False):

    """
   Function to balance (to min) the training data across different classes.

   Args:
    x_train (list): Input data for training.
    
    y_train (list): Labels corresponding to the input data. (one-hot encoded)
    
    dtype (cupy.dtype): Data type for the arrays. cp.float32 by default. Example: cp.float64 or cp.float16.

    shuffle_in_cpu (bool): If True, output will be same cpu's auto_balancer function. Default: False. (Use this for direct comparison of cpu training.)

   Returns:
    tuple: A tuple containing balanced input data and labels.
    """

    from .ui import loading_bars, get_loading_bar_style
    from .memory_operations import transfer_to_gpu
    
    x_train = transfer_to_gpu(x_train, dtype=dtype)
    y_train = transfer_to_gpu(y_train, dtype=y_train.dtype)

    bar_format = loading_bars()[0]
    
    classes = cp.arange(y_train.shape[1], dtype=y_train.dtype)
    class_count = len(classes)
    
   
    ClassIndices = {i: cp.where(y_train[:, i] == 1)[
        0] for i in range(class_count)}
    classes = [len(ClassIndices[i]) for i in range(class_count)]

    if len(set(classes)) == 1:
        print(Fore.WHITE + "INFO: Data have already balanced. from: auto_balancer" + Style.RESET_ALL)
        return x_train, y_train

    MinCount = min(classes)

    BalancedIndices = []
    for i in tqdm(range(class_count),leave=False, ascii=get_loading_bar_style(),
        bar_format= bar_format, desc='Balancing Data',ncols=70):
        if len(ClassIndices[i]) > MinCount:
            if shuffle_in_cpu:
                SelectedIndices = np.random.choice(
                    ClassIndices[i].get(), MinCount, replace=False)
            else:
                SelectedIndices = cp.random.choice(
                    ClassIndices[i], MinCount, replace=False)
        else:
            SelectedIndices = ClassIndices[i]
        BalancedIndices.extend(SelectedIndices)
            
    BalancedInputs = [x_train[idx] for idx in BalancedIndices]
    BalancedLabels = [y_train[idx] for idx in BalancedIndices]

    if shuffle_in_cpu:
        permutation = np.random.permutation(len(BalancedInputs))
    else:
        permutation = cp.random.permutation(len(BalancedInputs))

    BalancedInputs = cp.array(BalancedInputs)[permutation]
    BalancedLabels = cp.array(BalancedLabels)[permutation]

    print(Fore.GREEN + "Data Succesfully Balanced from: " + str(len(x_train)
                                                                                ) + " to: " + str(len(BalancedInputs)) + ". from: auto_balancer " + Style.RESET_ALL)
    del x_train, y_train
    cp.cuda.MemoryPool().free_all_blocks()

    return BalancedInputs, BalancedLabels


def synthetic_augmentation(x_train, y_train, dtype=cp.float32, shuffle_in_cpu=False):
    """
    Generates synthetic examples to balance classes with fewer examples using CuPy.
    Args:
        x_train: cupy array format

        y_train: cupy array format
        
        dtype (cupy.dtype): Data type for the arrays. cp.float32 by default. Example: cp.float64 or cp.float16.
        
        shuffle_in_cpu (bool): If True, output will be same cpu's synthetic_augmentation function. Default: False. (Use this for direct comparison of cpu training.)

    Returns:
        x_train_balanced, y_train_balanced (cupy array format)
    """
    from .ui import loading_bars, get_loading_bar_style
    from .memory_operations import transfer_to_gpu
    
    x = transfer_to_gpu(x_train, dtype=dtype)
    y = transfer_to_gpu(y_train, dtype=y_train.dtype)
    
    bar_format = loading_bars()[0]
    
    classes = cp.arange(y_train.shape[1])
    class_count = len(classes)
    class_distribution = {i: 0 for i in range(class_count)}
    
    for label in y:
        class_distribution[cp.argmax(label).item()] += 1
        
    max_class_count = max(class_distribution.values())
    x_balanced = list(x)
    y_balanced = list(y)
    
    for class_label in tqdm(range(class_count), leave=False, ascii=get_loading_bar_style(),
            bar_format=bar_format, desc='Augmenting Data', ncols=52):
        class_indices = [i for i, label in enumerate(y) if cp.argmax(label) == class_label]
        num_samples = len(class_indices)
        
        if num_samples < max_class_count:
            while num_samples < max_class_count:
                if shuffle_in_cpu:
                    random_indices = np.random.choice(
                        class_indices, 2, replace=False)
                else:
                    random_indices = cp.random.choice(
                    cp.array(class_indices), 2, replace=False)
                sample1 = x[random_indices[0]]
                sample2 = x[random_indices[1]]
                
                if shuffle_in_cpu:
                    synthetic_sample = sample1 + \
                        (sample2 - sample1) * np.random.rand()
                else:
                    synthetic_sample = sample1 + \
                        (sample2 - sample1) * cp.random.rand()
                
                x_balanced.append(synthetic_sample)
                y_balanced.append(y[class_indices[0]])
                num_samples += 1
    
    x_balanced = cp.array(x_balanced)
    y_balanced = cp.array(y_balanced)
    
    del x_train, y_train, x, y
    cp.cuda.MemoryPool().free_all_blocks()

    return x_balanced, y_balanced

def standard_scaler(x_train=None, x_test=None, scaler_params=None, dtype=cp.float32):
    """
    Standardizes training and test datasets. x_test may be None.

    Args:
        x_train (cupy.ndarray): 
        
        x_test (cupy.ndarray): (optional)
        
        scaler_params (tuple): (optional for using model)
        
        dtype (cupy.dtype): Data type for the arrays. cp.float32 by default. Example: cp.float64 or cp.float16.

    Returns:
        Scaler parameters, Standardized training (and if test given) datasets. (tuple)
    """
    if x_train is not None and scaler_params is None and x_test is not None:
        x_train = x_train.astype(dtype, copy=False)
        x_test = x_test.astype(dtype, copy=False)

        mean = cp.mean(x_train, axis=0)
        std = cp.std(x_train, axis=0)
        
        train_data_scaled = (x_train - mean) / std
        test_data_scaled = (x_test - mean) / std

        train_data_scaled = cp.nan_to_num(train_data_scaled, nan=0)
        test_data_scaled = cp.nan_to_num(test_data_scaled, nan=0)

        scaler_params = [mean, std]

        return scaler_params, train_data_scaled, test_data_scaled
    
    if scaler_params is None and x_train is None and x_test is not None:
        return x_test.astype(dtype, copy=False)  # sample data not scaled
            
    if scaler_params is not None:
        x_test = x_test.astype(dtype, copy=False)
        scaled_data = (x_test - scaler_params[0]) / scaler_params[1]
        scaled_data = cp.nan_to_num(scaled_data, nan=0)

        return scaled_data  # sample data scaled
    

def normalization(
    Input,  # num: Input data to be normalized.
    dtype=cp.float32
):
    """
    Normalizes the input data using maximum absolute scaling.

    Args:
        Input (num): Input data to be normalized.

        dtype (cupy.dtype): Data type for the arrays. cp.float32 by default. Example: cp.float64 or cp.float16.

    Returns:
        (num) Scaled input data after normalization.
    """

    MaxAbs = cp.max(cp.abs(Input.astype(dtype, copy=False)))
    return (Input / MaxAbs)


def non_neg_normalization(
    Input,
    dtype=cp.float32
):
    """
    Normalizes the input data [0-1] range.
    Args:
        Input (cupy): Input data to be normalized.
        dtype (cupy.dtype): Data type for the arrays. cp.float32 by default. Example: cp.float64 or cp.float16.
    Returns:
        (cupy) Scaled input data after normalization.
    """

    if isinstance(Input, np.ndarray):
        Input = cp.array(Input, dtype=dtype)

    Input = Input.astype(dtype, copy=False)
    MaxAbs = cp.max(cp.abs(Input))

    if cp.all(Input == Input.flat[0]):
        randomization = cp.random.random(Input.shape).astype(dtype)
        return randomization
    
    return (Input + MaxAbs) / (2 * MaxAbs)


def find_closest_factors(a):

    root = int(math.sqrt(a))
    
    for i in range(root, 0, -1):
        if a % i == 0:
            j = a // i
            return i, j
        
def batcher(x_test, y_test, batch_size=1):

    if batch_size == 1:
        return x_test, y_test

    y_labels = cp.argmax(y_test, axis=1)
    
    unique_labels = cp.unique(y_labels)
    total_samples = sum(
        int(cp.sum(y_labels == class_label) * batch_size) for class_label in unique_labels
    )
    
    sampled_x = cp.empty((total_samples, x_test.shape[1]), dtype=x_test.dtype)
    sampled_y = cp.empty((total_samples, y_test.shape[1]), dtype=y_test.dtype)
    
    offset = 0
    for class_label in unique_labels:
        class_indices = cp.where(y_labels == class_label)[0]
        
        num_samples = int(len(class_indices) * batch_size)
        
        sampled_indices = cp.random.choice(class_indices, num_samples, replace=False)
        
        sampled_x[offset:offset + num_samples] = x_test[sampled_indices]
        sampled_y[offset:offset + num_samples] = y_test[sampled_indices]

        offset += num_samples
    
    return sampled_x, sampled_y