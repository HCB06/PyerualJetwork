import numpy as np

def wals(acc, loss, acc_impact, loss_impact):
    """
    The WALS(weighted accuracy-loss score) function calculates a weighted sum of accuracy and loss based on their respective impacts.
    
    :param acc: The `acc` parameter represents the accuracy of a model or system
    :param loss: The `loss` parameter in the `wals` function represents the amount of loss incurred. It is used in the calculation to determine the overall impact based on the accuracy and loss impacts provided
    :param acc_impact: The `acc_impact` parameter represents the impact of accuracy on the overall score calculation in the `wals` function. It is a multiplier that determines how much the accuracy contributes to the final result
    :param loss_impact: The `loss_impact` parameter in the `wals` function represents the weight of loss value when calculating the overall impact. It is used to determine how much the loss affects the final result compared to the accuracy impact
    :return: the weighted sum of accuracy and loss based on their respective impacts.
    """
    loss += np.finfo(float).eps
    loss_impact += np.finfo(float).eps
    
    return (acc * acc_impact) + ((loss_impact / loss) * loss_impact)