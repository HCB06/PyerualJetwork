from pyerualjetwork import neu_cpu, data_operations_cpu, model_operations_cpu
import numpy as np
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt

# TRAIN

data = load_digits()

X = data.data
y = data.target

X = data_operations_cpu.normalization(X)

x_train, x_test, y_train, y_test = data_operations_cpu.split(X, y, 0.4, 42)

y_train, y_test = data_operations_cpu.encode_one_hot(y_train, y_test)
x_test, y_test = data_operations_cpu.auto_balancer(x_test, y_test)

W = neu_cpu.plan_fit(x_train, y_train)

# TEST

test_model = neu_cpu.evaluate(x_test, y_test, W=W)

model_operations_cpu.save_model(model_name='digits', W=W)

# REVERSE_PREDICT

output = [0,0,0,0,0,0,0,0,1,0]
Input = model_operations_cpu.reverse_predict_from_memory(output, W)

plt.imshow(np.reshape(Input, (8,8)))
plt.show()