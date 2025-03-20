# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 23:32:16 2024

@author: hasan
"""

from pyerualjetwork import neu_cpu, ene_cpu, data_operations_cpu, model_operations_cpu, metrics_cpu
from sklearn.datasets import load_wine

data = load_wine()
X = data.data
y = data.target

x_train, x_test, y_train, y_test = data_operations_cpu.split(X, y, 0.4, 42)

y_train, y_test = data_operations_cpu.encode_one_hot(y_train, y_test)

x_test, y_test = data_operations_cpu.auto_balancer(x_test, y_test)

scaler_params, x_train, x_test = data_operations_cpu.standard_scaler(x_train, x_test)

# Configuring optimizer
genetic_optimizer = lambda *args, **kwargs: ene_cpu.evolver(*args, **kwargs)
model = neu_cpu.learn(x_train, y_train, genetic_optimizer, fit_start=True, show_history=True, gen=2)

W = model[model_operations_cpu.get_weights()]
activation_potentiation = model[model_operations_cpu.get_act()]

test_model = neu_cpu.evaluate(x_test, y_test, show_metrics_cpu=True, W=W, activations=activation_potentiation)

test_preds = test_model[model_operations_cpu.get_preds()]
test_acc = test_model[model_operations_cpu.get_acc()]

model_name = 'wine'
model_path = ''

model_operations_cpu.save_model(model_name=model_name, model_type='PLAN', activations=model[model_operations_cpu.get_act()], model_path=model_path, scaler_params=scaler_params, W=model[model_operations_cpu.get_weights()])

precisison, recall, f1 = metrics_cpu.metrics_cpu(y_test, test_preds)
print('Precision: ', precisison, '\n', 'Recall: ', recall, '\n', 'F1: ', f1)
