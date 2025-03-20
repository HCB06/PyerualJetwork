import time
from colorama import Fore
from pyerualjetwork import neu, ene, data_operations, model_operations, metrics
from sklearn.datasets import load_iris
import numpy as np

data = load_iris()
X = data.data
y = data.target

x_train, x_test, y_train, y_test = data_operations.split(X, y, 0.3, 42)

y_train, y_test = data_operations.encode_one_hot(y_train, y_test)

x_train, y_train = data_operations.synthetic_augmentation(x_train, y_train)

scaler_params, x_train, x_test = data_operations.standard_scaler(x_train, x_test)

# Configuring optimizer
genetic_optimizer = lambda *args, **kwargs: ene.evolver(*args, activation_mutate_add_prob=0, activation_selection_add_prob=0, **kwargs)

model = neu.learn(x_train, y_train, optimizer=genetic_optimizer, fit_start=True, gen=50, target_acc=0.96, neural_web_history=True, interval=16.67)

W = model[model_operations.get_weights()]

test_model = neu.evaluate(x_test, y_test, W=W, activations=model[model_operations.get_act()])

test_preds = test_model[model_operations.get_preds()]
test_acc = test_model[model_operations.get_acc()]

model_operations.save_model(model_name='iris',
                 model_type='PLAN',
                 test_acc=test_acc,
                 weights_type='npy',
                 weights_format='raw',
                 model_path='',
                 activations=model[model_operations.get_act()],
                 scaler_params=scaler_params,
                 show_architecture=True,
                 W=W)

precisison, recall, f1 = metrics.metrics(y_test, data_operations.decode_one_hot(test_preds))
print('Precision: ', precisison, '\n', 'Recall: ', recall, '\n', 'F1: ', f1)

y_test = data_operations.decode_one_hot(y_test)


for i in range(len(x_test)):
    Predict = model_operations.predict_from_memory(x_test[i], W=W, activations=model[model_operations.get_act()])
    time.sleep(0.6)
    if np.argmax(Predict) == y_test[i]:
        print(Fore.GREEN + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', y_test[i])
    else:
        print(Fore.RED + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', y_test[i])
