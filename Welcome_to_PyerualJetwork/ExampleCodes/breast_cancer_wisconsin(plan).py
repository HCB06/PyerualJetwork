# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 02:36:20 2024

@author: hasan
"""
from colorama import Fore
from sklearn.datasets import load_breast_cancer
from pyerualjetwork.cpu import data_ops
from pyerualjetwork import nn, ene, model_ops
import numpy as np
import time

# Breast Cancer veri setini yükleme
data = load_breast_cancer()
X = data.data
y = data.target

# Eğitim, test ve doğrulama verilerini ayırma
x_train, x_test, y_train, y_test = data_ops.split(X, y, 0.4, 42)

x_train, x_val, y_train, y_val = data_ops.split(x_train, y_train, 0.2, 42)

# One-hot encoding işlemi
y_train, y_test = data_ops.encode_one_hot(y_train, y_test)
y_val = data_ops.encode_one_hot(y_val, y)[0]

x_train, y_train = data_ops.auto_balancer(x_train, y_train)
x_test, y_test = data_ops.auto_balancer(x_test, y_test)

scaler_params, x_train, x_test = data_ops.standard_scaler(x_train, x_test)

# Modeli eğitme
W = nn.plan_fit(x_train, y_train)

# Modeli test etme
test_results = nn.evaluate(x_test, y_test, W=W, model_type='PLAN')
print(f"Test Accuracy: {test_results[model_ops.get_acc()]}")

# Modeli build etme
model = model_ops.build_model(W, activations=['linear'], model_type='PLAN')

# Scaler parametrelerini modele yaz ve artık model kaydedilmeye hazır.
model = model._replace(scaler_params=scaler_params)

# Modeli kaydetme
model_ops.save_model(model, model_name='breast_cancer')

# Model tahminlerini değerlendirme
for i in range(len(x_val)):
    Predict = model_ops.predict_from_storage(model_name='breast_cancer', model_path='', Input=x_val[i])

    time.sleep(0.5)
    if np.argmax(Predict) == np.argmax(y_val[i]):
        print(Fore.GREEN + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', np.argmax(y_val[i]))
    else:
        print(Fore.RED + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', np.argmax(y_val[i]))