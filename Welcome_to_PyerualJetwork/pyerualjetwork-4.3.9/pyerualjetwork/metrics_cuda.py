import cupy as cp

def metrics(y_ts, test_preds, average='weighted'):
    from .data_operations import decode_one_hot
    y_test_d = cp.array(decode_one_hot(y_ts))
    y_pred = cp.array(test_preds)

    if y_test_d.ndim > 1:
        y_test_d = y_test_d.ravel()
    if y_pred.ndim > 1:
        y_pred = y_pred.ravel()

    classes = cp.unique(cp.concatenate((y_test_d, y_pred)))
    tp = cp.zeros(len(classes), dtype=cp.int32)
    fp = cp.zeros(len(classes), dtype=cp.int32)
    fn = cp.zeros(len(classes), dtype=cp.int32)

    for i, c in enumerate(classes):
        tp[i] = cp.sum((y_test_d == c) & (y_pred == c))
        fp[i] = cp.sum((y_test_d != c) & (y_pred == c))
        fn[i] = cp.sum((y_test_d == c) & (y_pred != c))

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    if average == 'micro':
        tp_sum = cp.sum(tp)
        fp_sum = cp.sum(fp)
        fn_sum = cp.sum(fn)
        precision_val = tp_sum / (tp_sum + fp_sum + 1e-10)
        recall_val = tp_sum / (tp_sum + fn_sum + 1e-10)
        f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val + 1e-10)

    elif average == 'macro':
        precision_val = cp.mean(precision)
        recall_val = cp.mean(recall)
        f1_val = cp.mean(f1)

    elif average == 'weighted':
        weights = cp.array([cp.sum(y_test_d == c) for c in classes])
        weights = weights / cp.sum(weights)
        precision_val = cp.sum(weights * precision)
        recall_val = cp.sum(weights * recall)
        f1_val = cp.sum(weights * f1)

    else:
        raise ValueError("Invalid value for 'average'. Choose from 'micro', 'macro', 'weighted'.")

    return precision_val.item(), recall_val.item(), f1_val.item()


def roc_curve(y_true, y_score):
    """
    Compute Receiver Operating Characteristic (ROC) curve.

    Parameters:
    y_true : array, shape = [n_samples]
        True binary labels in range {0, 1} or {-1, 1}.
    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned
        by decision_function on some classifiers).

    Returns:
    fpr : array, shape = [n]
        Increasing false positive rates such that element i is the false positive rate
        of predictions with score >= thresholds[i].
    tpr : array, shape = [n]
        Increasing true positive rates such that element i is the true positive rate
        of predictions with score >= thresholds[i].
    thresholds : array, shape = [n]
        Decreasing thresholds on the decision function used to compute fpr and tpr.
    """
    
    y_true = cp.asarray(y_true)
    y_score = cp.asarray(y_score)

    if len(cp.unique(y_true)) != 2:
        raise ValueError("Only binary classification is supported.")

    
    desc_score_indices = cp.argsort(y_score, kind="stable")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]


    fpr = []
    tpr = []
    thresholds = []
    n_pos = cp.sum(y_true)
    n_neg = len(y_true) - n_pos

    tp = 0
    fp = 0
    prev_score = 0

    for i, score in enumerate(y_score):
        if score is not prev_score:
            fpr.append(fp / n_neg)
            tpr.append(tp / n_pos)
            thresholds.append(score)
            prev_score = score

        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1

    fpr.append(fp / n_neg)
    tpr.append(tp / n_pos)
    thresholds.append(score)

    return cp.array(fpr), cp.array(tpr), cp.array(thresholds)


def confusion_matrix(y_true, y_pred, class_count):
    """
    Computes confusion matrix.

    Args:
        y_true (numpy.ndarray): True class labels (1D array).
        y_pred (numpy.ndarray): Predicted class labels (1D array).
        num_classes (int): Number of classes.

    Returns:
        numpy.ndarray: Confusion matrix of shape (num_classes, num_classes).
    """
    confusion = cp.zeros((class_count, class_count), dtype=int)

    for i in range(len(y_true)):
        true_label = y_true[i]
        pred_label = y_pred[i]
        confusion[true_label, pred_label] += 1

    return confusion


def pca(X, n_components):
    """
    
    Parameters:
    X (numpy array): (n_samples, n_features)
    n_components (int):
    
    Returns:
    X_reduced (numpy array): (n_samples, n_components)
    """
    
    X_meaned = X - cp.mean(X, axis=0)
    
    covariance_matrix = cp.cov(X_meaned, rowvar=False)
    
    eigenvalues, eigenvectors = cp.linalg.eigh(covariance_matrix)
    
    sorted_index = cp.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_index]
    
    eigenvectors_subset = sorted_eigenvectors[:, :n_components]
    
    X_reduced = cp.dot(X_meaned, eigenvectors_subset)
    
    return X_reduced