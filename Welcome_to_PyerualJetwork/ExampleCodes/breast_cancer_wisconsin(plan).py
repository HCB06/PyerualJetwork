# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 02:36:20 2024

@author: hasan
"""
from colorama import Fore
from sklearn.datasets import load_breast_cancer
from pyerualjetwork import neu_cpu, data_operations_cpu, model_operations_cpu
import numpy as np
import time

# Breast Cancer veri setini yükleme
data = load_breast_cancer()
X = data.data
y = data.target

# Eğitim, test ve doğrulama verilerini ayırma
x_train, x_test, y_train, y_test = data_operations_cpu.split(X, y, 0.4, 42)

x_train, x_val, y_train, y_val = data_operations_cpu.split(x_train, y_train, 0.2, 42)

# One-hot encoding işlemi
y_train, y_test = data_operations_cpu.encode_one_hot(y_train, y_test)
y_val = data_operations_cpu.encode_one_hot(y_val, y)[0]

x_train, y_train = data_operations_cpu.auto_balancer(x_train, y_train)
x_test, y_test = data_operations_cpu.auto_balancer(x_test, y_test)

scaler_params, x_train, x_test = data_operations_cpu.standard_scaler(x_train, x_test)

# Modeli eğitme
W = neu_cpu.plan_fit(x_train, y_train)

# Modeli test etme
test_model = neu_cpu.evaluate(x_test, y_test, W=W)
print(f"Test Accuracy: {test_model[model_operations_cpu.get_acc()]}")

# Modeli kaydetme
model_operations_cpu.save_model(model_name='breast_cancer',
                test_acc=test_model[model_operations_cpu.get_acc()],
                scaler_params=scaler_params, model_type='PLAN',
                W=W)

# Model tahminlerini değerlendirme
for i in range(len(x_val)):
    Predict = model_operations_cpu.predict_from_storage(model_name='breast_cancer', model_path='', Input=x_val[i])

    time.sleep(0.5)
    if np.argmax(Predict) == np.argmax(y_val[i]):
        print(Fore.GREEN + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', np.argmax(y_val[i]))
    else:
        print(Fore.RED + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', np.argmax(y_val[i]))
