from pyerualjetwork import neu_cpu, data_operations_cpu, model_operations_cpu
import time
from colorama import Fore
import numpy as np
from sklearn.datasets import load_digits

# TRAIN

data = load_digits()

X = data.data
y = data.target

X = data_operations_cpu.normalization(X)

x_train, x_test, y_train, y_test = data_operations_cpu.split(X, y, 0.4, 42)

y_train, y_test = data_operations_cpu.encode_one_hot(y_train, y_test)


x_test, y_test = data_operations_cpu.auto_balancer(x_test, y_test)

activation_potentiation=['bent_identity']

W = neu_cpu.plan_fit(x_train, y_train, activations=activation_potentiation)

# TEST

test_model = neu_cpu.evaluate(x_test, y_test, show_metrics=True, W=W, activations=activation_potentiation)
test_preds = test_model[model_operations_cpu.get_preds()]
test_acc = test_model[model_operations_cpu.get_acc()]

# PREDICT

for i in range(len(x_test)):
    Predict = model_operations_cpu.predict_from_memory(x_test[i], W=W, activations=activation_potentiation)

    time.sleep(0.5)
    if np.argmax(Predict) == np.argmax(y_test[i]):
        print(Fore.GREEN + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', np.argmax(y_test[i]))
    else:
        print(Fore.RED + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', np.argmax(y_test[i]))
