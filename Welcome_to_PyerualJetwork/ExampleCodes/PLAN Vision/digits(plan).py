from pyerualjetwork import model_ops, nn
from pyerualjetwork.cpu import data_ops
import time
from colorama import Fore
import numpy as np
from sklearn.datasets import load_digits

# TRAIN

data = load_digits()

X = data.data
y = data.target

X = data_ops.normalization(X)

x_train, x_test, y_train, y_test = data_ops.split(X, y, 0.4, 42)

y_train, y_test = data_ops.encode_one_hot(y_train, y_test)


x_test, y_test = data_ops.auto_balancer(x_test, y_test)

activation_potentiation=['bent_identity']

W = nn.plan_fit(x_train, y_train, activations=activation_potentiation)

model = model_ops.build_model(W, activations=activation_potentiation, model_type='PLAN')

# TEST

test_results = nn.evaluate(x_test, y_test, model)
test_preds = test_results[model_ops.get_preds()]
test_acc = test_results[model_ops.get_acc()]

# PREDICT

for i in range(len(x_test)):
    Predict = model_ops.predict_from_memory(x_test[i], model)

    time.sleep(0.5)
    if np.argmax(Predict) == np.argmax(y_test[i]):
        print(Fore.GREEN + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', np.argmax(y_test[i]))
    else:
        print(Fore.RED + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', np.argmax(y_test[i]))
