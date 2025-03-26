from pyerualjetwork.cpu import nn, data_ops, model_ops
import numpy as np
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt

# TRAIN

data = load_digits()

X = data.data
y = data.target

X = data_ops.normalization(X)

x_train, x_test, y_train, y_test = data_ops.split(X, y, 0.4, 42)

y_train, y_test = data_ops.encode_one_hot(y_train, y_test)
x_test, y_test = data_ops.auto_balancer(x_test, y_test)

W = nn.plan_fit(x_train, y_train)

# TEST

test_model = nn.evaluate(x_test, y_test, model_type='PLAN', W=W)

model_ops.save_model(model_name='digits', model_type='PLAN', W=W)

# REVERSE_PREDICT

output = [0,0,0,0,0,0,0,0,1,0]
Input = model_ops.reverse_predict_from_memory(output, W)

plt.imshow(np.reshape(Input, (8,8)))
plt.show()