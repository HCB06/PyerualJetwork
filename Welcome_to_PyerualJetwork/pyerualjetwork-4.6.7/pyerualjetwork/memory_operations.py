import psutil
import numpy as np
import cupy as cp
import logging

def get_available_cpu_memory():
    """
    The function `get_available_memory` returns the amount of available memory in the system using the
    `psutil` library.
    :return: The function `get_available_memory()` returns the amount of available memory in bytes on
    the system.
    """
    return psutil.virtual_memory().available

def get_optimal_batch_size_for_cpu(x, data_size_bytes, available_memory):
    """
    The function calculates the optimal batch size for a given data size and available memory based on
    the size of each element.
    
    :param x: `x` is a NumPy array representing the input data for which we want to determine the optimal batch size for processing on the CPU
    
    :param data_size_bytes: The `data_size_bytes` parameter represents the size of the data in bytes that you want to process in batches
    
    :param available_memory: The `available_memory` parameter represents the total memory available on the CPU in bytes. This function calculates the optimal batch size for processing data based on the provided parameters. Let me know if you need any further assistance or explanation!
    
    :return: the optimal batch size for a given array `x` based on the available memory and the size of each element in bytes.
    """
    safe_memory = available_memory * 0.25
    element_size = data_size_bytes / x.size
    return int(safe_memory / (element_size * 2))

def transfer_to_cpu(x, dtype=np.float32):
    """
    The `transfer_to_cpu` function converts data to a specified data type on the CPU, handling memory constraints
    by batching the conversion process and ensuring complete GPU memory cleanup.
   
    :param x: Input data to transfer to CPU (CuPy array)
    
    :param dtype: Target NumPy dtype for the output array (default: np.float32)
    
    :return: NumPy array with the specified dtype
    """
    from .ui import loading_bars, initialize_loading_bar
    try:
        if isinstance(x, np.ndarray):
            return x.astype(dtype) if x.dtype != dtype else x

        x = x.astype(dtype=dtype, copy=False)
        
        data_size = x.nbytes
        available_memory = get_available_cpu_memory()
        logging.debug(f"Data size: {data_size/1e6:.2f}MB, Available memory: {available_memory/1e6:.2f}MB")
        
        pool = cp.get_default_memory_pool()
        pinned_mempool = cp.cuda.PinnedMemoryPool()
        
        if data_size <= available_memory * 0.25:
            try:
                final_result = np.array(x.get(), dtype=dtype, copy=False)
            finally:
                del x
                pool.free_all_blocks()
                pinned_mempool.free_all_blocks()
                cp.cuda.runtime.deviceSynchronize()
            return final_result
        
        batch_size = max(get_optimal_batch_size_for_cpu(x, data_size, available_memory), 1)
        total_batches = (len(x) + batch_size - 1) // batch_size
        loading_bar = initialize_loading_bar(
            total=total_batches,
            desc='Transfering to CPU mem',
            ncols=70,
            bar_format=loading_bars()[0],
            leave=False
        )
        logging.debug(f"Using batch size: {batch_size}")
        
        try:
            sample_chunk = x[0:1]
            sample_array = np.array(sample_chunk.get(), dtype=dtype)
            chunk_shape = sample_array.shape[1:] if len(sample_array.shape) > 1 else ()
            total_shape = (len(x),) + chunk_shape
        finally:
            del sample_array
            del sample_chunk
            pool.free_all_blocks()
            pinned_mempool.free_all_blocks()
        
        chunks = np.empty(total_shape, dtype=dtype)
        
        try:
            for i in range(0, len(x), batch_size):
                try:
                    end_idx = min(i + batch_size, len(x))
                    chunk = x[i:end_idx]
                    chunks[i:end_idx] = chunk.get().astype(dtype=dtype)
                finally:
                    del chunk
                    pool.free_all_blocks()
                    pinned_mempool.free_all_blocks()
                    cp.cuda.runtime.deviceSynchronize()
                
                loading_bar.update(1)
        finally:
            del x
            pool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            cp.cuda.runtime.deviceSynchronize()
        
        return chunks
        
    except Exception as e:
        logging.error(f"Error in transfer_to_cpu: {str(e)}")
        if 'x' in locals():
            del x
        if 'pool' in locals():
            pool.free_all_blocks()
        if 'pinned_mempool' in locals():
            pinned_mempool.free_all_blocks()
        cp.cuda.runtime.deviceSynchronize()
        raise

def get_optimal_batch_size_for_gpu(x, data_size_bytes):
    """
    The function calculates the optimal batch size for a GPU based on available memory and data size.
    
    :param x: A list or array containing the data elements that will be processed on the GPU
    :param data_size_bytes: The `data_size_bytes` parameter represents the total size of the data in bytes that you want to process on the GPU. This could be the size of a single batch of data or the total size of the dataset, depending on how you are structuring your computations
    :return: the optimal batch size that can be used for processing the given data on the GPU, based on the available free memory on the GPU and the size of the data elements.
    """
    free_memory = cp.get_default_memory_pool().free_bytes()
    device_memory = cp.cuda.runtime.memGetInfo()[0]
    safe_memory = min(free_memory, device_memory) * 0.25
    
    element_size = data_size_bytes / len(x)
    return int(safe_memory / (element_size * 2))


def transfer_to_gpu(x, dtype=cp.float32):
    """
    The `transfer_to_gpu` function in Python converts input data to GPU arrays, optimizing memory usage by
    batching and handling out-of-memory errors.
    
    :param x: The `x` parameter in the `transfer_to_gpu` function is the input data that you want to transfer to the GPU for processing. It can be either a NumPy array or a CuPy array. If it's a NumPy array, the function will convert it to a CuPy array and
    
    :param dtype: The `dtype` parameter in the `transfer_to_gpu` function specifies the data type to which the input array `x` should be converted when moving it to the GPU. By default, it is set to `cp.float32`, which is a 32-bit floating-point data type provided by the CuPy
    
    :return: The `transfer_to_gpu` function returns the input data `x` converted to a GPU array of type `dtype` (default is `cp.float32`). If the input `x` is already a GPU array with the same dtype, it returns `x` as is. If the data size of `x` exceeds 25% of the free GPU memory, it processes the data in batches to
    """
    from .ui import loading_bars, initialize_loading_bar
    try:
        if isinstance(x, cp.ndarray):
            return x.astype(dtype) if x.dtype != dtype else x
            
        x = x.astype(dtype=dtype, copy=False)
        data_size = x.nbytes
        pinned_mempool = cp.cuda.PinnedMemoryPool()
        free_gpu_memory = cp.cuda.runtime.memGetInfo()[0]
        logging.debug(f"Data size: {data_size/1e6:.2f}MB, Free GPU memory: {free_gpu_memory/1e6:.2f}MB")

        if data_size <= free_gpu_memory * 0.25:
            new_x = cp.array(x, dtype=dtype, copy=False)
            return new_x
            
        batch_size = get_optimal_batch_size_for_gpu(x, data_size)
        if batch_size == 0: batch_size = 1

        loading_bar = initialize_loading_bar(total=len(x)/batch_size, desc='Transfering to GPU mem', ncols=70, bar_format=loading_bars()[0], leave=False)
        
        logging.debug(f"Using batch size: {batch_size}")
        current_threshold = 0.75
        total_batches = (len(x) + batch_size - 1) // batch_size

        sample_chunk = x[0:1]
        sample_array = cp.array(sample_chunk, dtype=dtype)
        chunk_shape = sample_array.shape[1:] if len(sample_array.shape) > 1 else ()
        del sample_array
        del sample_chunk
        if chunk_shape:
            total_shape = (len(x),) + chunk_shape
        else:
            total_shape = (len(x),)
        
        del chunk_shape
        chunks = cp.empty(total_shape, dtype=dtype)
        del total_shape

        for i in range(0, len(x), batch_size):
            try:
                chunk = x[i:i + batch_size]
                chunk = cp.array(chunk, dtype=dtype)
                chunks[i // batch_size] = chunk
                del chunk
                pinned_mempool.free_all_blocks()

                if i > 0 and i % (batch_size * 5) == 0:
                    pool = cp.get_default_memory_pool()
                    current_threshold = adjust_gpu_memory_threshold(pool, free_gpu_memory, current_threshold)
                    if pool.used_bytes() > cp.cuda.runtime.memGetInfo()[0] * current_threshold:
                        pool.free_all_blocks()
                    
                    
                loading_bar.update(1)
                    
            except cp.cuda.memory.OutOfMemoryError:
                logging.error(f"GPU out of memory at batch {i//batch_size + 1}/{total_batches}")
                cp.get_default_memory_pool().free_all_blocks()
                batch_size = max(batch_size // 2, 1)
                continue
                
            except Exception as e:
                logging.error(f"Error processing batch {i//batch_size + 1}/{total_batches}: {str(e)}")
                raise
                
        try:
            del x
            cp.get_default_memory_pool().free_all_blocks()
            pinned_mempool.free_all_blocks()
            return chunks
            
        except Exception as e:
            logging.error(f"Error concatenating results: {str(e)}")
            raise
            
    except Exception as e:
        logging.error(f"Error in transfer_to_gpu: {str(e)}")
        raise

def adjust_gpu_memory_threshold(pool, free_gpu_memory, current_threshold=0.75, min_threshold=0.5, max_threshold=0.9):
    used_memory = pool.used_bytes()
    usage_ratio = used_memory / free_gpu_memory
    
    if usage_ratio > current_threshold:
        current_threshold = max(min_threshold, current_threshold - 0.05)
    elif usage_ratio < current_threshold * 0.8:
        current_threshold = min(max_threshold, current_threshold + 0.05)

    return current_threshold


def optimize_labels(y, one_hot_encoded=True, cuda=False):
    """
    The function `optimize_labels` optimizes the data type of labels based on their length and encoding
    format.
    
    :param y: The `optimize_labels` function is designed to optimize the data type of the input labels `y` based on certain conditions. The function checks if the labels are in one-hot encoded format or not, and then based on the length of the labels and the specified data types (`uint8`, `uint
    :param one_hot_encoded: The `one_hot_encoded` parameter in the `optimize_labels` function indicates whether the labels are in one-hot encoded format or not. If `one_hot_encoded` is set to `True`, it means that the labels are in one-hot encoded format, and the function will check the length of the, defaults to True (optional)
    :param cuda: The `cuda` parameter in the `optimize_labels` function is a boolean flag that indicates whether to use CUDA for computations. If `cuda` is set to `True`, the function will use the CuPy library for array operations, which can leverage GPU acceleration. If `cuda` is `False, defaults to False (optional)
    :return: The function `optimize_labels` returns the input array `y` after optimizing its data type based on the specified conditions. If `one_hot_encoded` is True, it checks the length of the elements in `y` and converts the data type to uint8, uint16, or uint32 accordingly. If `one_hot_encoded` is False, it checks the length of `y` itself and
    """

    if cuda: array_type = cp
    else: array_type = np

    dtype_uint8 = array_type.uint8
    dtype_uint16 = array_type.uint16
    dtype_uint32 = array_type.uint32

    if one_hot_encoded:
        if len(y[0]) < 256:
            if y.dtype != dtype_uint8:
                y = array_type.array(y, copy=False).astype(dtype_uint8, copy=False)
        elif len(y[0]) <= 32767:
            if y.dtype != dtype_uint16:
                y = array_type.array(y, copy=False).astype(dtype_uint16, copy=False)
        else:
            if y.dtype != dtype_uint32:
                y = array_type.array(y, copy=False).astype(dtype_uint32, copy=False)

        return y

    else:

        if len(y) < 256:
            if y.dtype != dtype_uint8:
                y = array_type.array(y, copy=False).astype(dtype_uint8, copy=False)
        elif len(y) <= 32767:
            if y.dtype != dtype_uint16:
                y = array_type.array(y, copy=False).astype(dtype_uint16, copy=False)
        else:
            if y.dtype != dtype_uint32:
                y = array_type.array(y, copy=False).astype(dtype_uint32, copy=False)

        return y