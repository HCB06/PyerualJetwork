import time
from colorama import Fore
from pyerualjetwork.cpu import data_ops, metrics
from pyerualjetwork import nn, ene, model_ops
from sklearn.datasets import load_iris
import numpy as np

data = load_iris()
X = data.data
y = data.target

x_train, x_test, y_train, y_test = data_ops.split(X, y, 0.3, 42)

y_train, y_test = data_ops.encode_one_hot(y_train, y_test)

x_train, y_train = data_ops.synthetic_augmentation(x_train, y_train)

scaler_params, x_train, x_test = data_ops.standard_scaler(x_train, x_test)

# Configuring optimizer
genetic_optimizer = lambda *args, **kwargs: ene.evolver(*args, strategy='more_selective', **kwargs)

model = nn.learn(x_train, 
                 y_train, 
                 optimizer=genetic_optimizer,
                 neurons=[16, 16], 
                 activation_functions=['tanh', 'tanh'], 
                 template_model=model_ops.get_model_template(), 
                 fit_start=False,
                 gen=50, 
                 pop_size=500)

test_results = nn.evaluate(x_test, y_test, model, show_report=True)

test_preds = test_results[model_ops.get_preds()]

precisison, recall, f1 = metrics.metrics(y_test, data_ops.decode_one_hot(test_preds))
print('Precision: ', precisison, '\n', 'Recall: ', recall, '\n', 'F1: ', f1)

y_test = data_ops.decode_one_hot(y_test)

for i in range(len(x_test)):
    Predict = model_ops.predict_from_memory(x_test[i], model)
    time.sleep(0.6)
    if np.argmax(Predict) == y_test[i]:
        print(Fore.GREEN + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', y_test[i])
    else:
        print(Fore.RED + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', y_test[i])


# Scaler parametrelerini modele yaz ve artık model kaydedilmeye hazır.
model = model._replace(scaler_params=scaler_params)

model_ops.save_model(model, model_name='iris', weights_format='raw', weights_type='npy')
