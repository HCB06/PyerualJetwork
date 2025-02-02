import cupy as cp

def cupy_pairwise_distances(X):
    """
    Pairwise Euclidean distances
    """
    r = cp.sum(X * X, 1)
    r = r.reshape(-1, 1)

    distances = cp.maximum(r - 2 * cp.dot(X, X.T) + r.T, 0.0)

    triu_indices = cp.triu_indices(distances.shape[0], k=1)
    return cp.sqrt(distances[triu_indices])


def diversity_score(population):
    """
    Calculates the diversity score of a population based on the
    Euclidean distances between individuals in the population.
   
    :param population: That calculates the diversity score of a population based on the Euclidean distances between
    individuals in the population
    :return: The function returns the diversity score,
    which is a measure of how spread out or diverse the population is in terms of their characteristics.
    """
    if len(population) < 2:
        return 0
    
    population = cp.nan_to_num(population, nan=0.0)
    
    distances = cupy_pairwise_distances(population)
    distances = cp.maximum(distances, 1e-10)
    
    avg_distance = cp.mean(distances)
    if population.shape[1] == 0:
        return 0
    
    max_possible_distance = cp.sqrt(population.shape[1])
    max_possible_distance = float(max(max_possible_distance, 1e-10))
    
    diversity = float(avg_distance / max_possible_distance)
    return diversity


def multi_head_fitness(y_true, y_pred, diversity_score, accuracy, alpha=2, beta=1.5, lambda_div=0.05):
    """
    The function calculates a fitness score based on accuracy, margin loss, and diversity score using
    specified parameters.
    
    @param y_true The `y_true` parameter represents the true labels of the data points. It is an array
    or list containing the actual labels of the data points.
    @param y_pred The `y_pred` parameter in the `multi_head_fitness` function represents the
    predicted values for a given dataset. It is a NumPy array containing the predicted values for each
    sample in the dataset.
    @param diversity_score The `diversity_score` parameter in the `multi_head_fitness` function
    represents a measure of diversity in the predictions. It is used as a factor in calculating the
    fitness of the model. The function combines accuracy, margin loss, and diversity score to evaluate
    the overall performance of the model
    @param accuracy Accuracy is a measure of the correct predictions made by a model. It is typically
    calculated as the number of correct predictions divided by the total number of predictions. In the
    context of the `multi_head_fitness` function, the accuracy parameter represents the accuracy
    of the model's predictions.
    @param alpha Alpha is a parameter that controls the impact of accuracy on the overall fitness score
    in the multi_head_fitness function. It is used as an exponent to raise the accuracy value
    to, influencing its contribution to the fitness calculation.
    @param beta The `beta` parameter in the `multi_head_fitness` function is used as an exponent
    in the fitness calculation formula. It controls the impact of the margin loss term on the overall
    fitness value. A higher value of `beta` will amplify the effect of the margin loss term, making it
    @param lambda_div The `lambda_div` parameter in the `multi_head_fitness` function represents
    the weight given to the diversity score in calculating the fitness value. It is a hyperparameter
    that controls the impact of diversity score on the overall fitness calculation. A higher value of
    `lambda_div` will increase the importance
    @return The function `multi_head_fitness` returns the fitness value calculated based on the
    icput parameters `y_true`, `y_pred`, `diversity_score`, `accuracy`, and optional parameters `alpha`,
    `beta`, and `lambda_div`. The fitness value is computed using a formula that combines accuracy,
    margin loss, and diversity score with specified weights and coefficients.
    """
    incorrect = y_true != y_pred
    margin_loss = cp.abs(y_true[incorrect] - y_pred[incorrect]).mean() if cp.any(incorrect) else 0

    margin_loss = cp.nan_to_num(margin_loss, nan=0.0)
    accuracy = max(accuracy, 1e-10)
    
    fitness = (accuracy ** alpha) * ((1 - margin_loss) ** beta) * (1 + lambda_div * diversity_score)
    return fitness