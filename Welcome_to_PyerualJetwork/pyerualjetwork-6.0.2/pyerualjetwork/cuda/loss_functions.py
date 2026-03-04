
import cupy as cp

def categorical_crossentropy(y_true_batch, y_pred_batch):
    epsilon = 1e-7
    y_pred_batch = cp.clip(y_pred_batch, epsilon, 1. - epsilon)
    
    losses = -cp.sum(y_true_batch * cp.log(y_pred_batch), axis=1)

    mean_loss = cp.mean(losses)
    return mean_loss


def binary_crossentropy(y_true_batch, y_pred_batch):
    epsilon = 1e-7
    y_pred_batch = cp.clip(y_pred_batch, epsilon, 1. - epsilon)
    
    losses = -cp.mean(y_true_batch * cp.log(y_pred_batch) + (1 - y_true_batch) * cp.log(1 - y_pred_batch), axis=1)

    mean_loss = cp.mean(losses)
    return mean_loss