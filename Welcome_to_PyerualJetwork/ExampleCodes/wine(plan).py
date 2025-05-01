# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 23:32:16 2024

@author: hasan
"""

from pyerualjetwork.cpu import data_ops, metrics
from pyerualjetwork import nn, ene, model_ops
from sklearn.datasets import load_wine

data = load_wine()
X = data.data
y = data.target

x_train, x_test, y_train, y_test = data_ops.split(X, y, 0.4, 42)

y_train, y_test = data_ops.encode_one_hot(y_train, y_test)

x_test, y_test = data_ops.auto_balancer(x_test, y_test)

scaler_params, x_train, x_test = data_ops.standard_scaler(x_train, x_test)

# Configuring optimizer
genetic_optimizer = lambda *args, **kwargs: ene.evolver(*args, **kwargs)
model = nn.learn(x_train, y_train, genetic_optimizer, model_ops.get_model_template(), fit_start=True, show_history=True, gen=2, pop_size=100)

test_results = nn.evaluate(x_test, y_test, model, show_report=True)

test_preds = test_results[model_ops.get_preds()]
test_acc = test_results[model_ops.get_acc()]

model_name = 'wine'
model_path = ''

model_ops.save_model(model, model_name=model_name, model_path=model_path)

precisison, recall, f1 = metrics.metrics(y_test, data_ops.decode_one_hot(test_preds))
print('Precision: ', precisison, '\n', 'Recall: ', recall, '\n', 'F1: ', f1)