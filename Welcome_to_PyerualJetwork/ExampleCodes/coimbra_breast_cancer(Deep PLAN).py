import pandas as pd
import numpy as np
from colorama import Fore
from pyerualjetwork.cpu import nn, ene, data_ops, model_ops
import time
from sklearn.metrics import classification_report

file_path = 'breast_cancer_coimbra.csv' 
data = pd.read_csv(file_path)


X = data.drop('Classification', axis=1).values
y = data['Classification'].values

# Eğitim, test ve doğrulama verilerini ayırma
x_train, x_test, y_train, y_test = data_ops.split(X, y, 0.4, 42)
x_train, x_val, y_train, y_val = data_ops.split(x_train, y_train, 0.2, 42)

# One-hot encoding işlemi
y_train, y_test = data_ops.encode_one_hot(y_train, y_test)
y_val = data_ops.encode_one_hot(y_val, y)[0]

# Veri dengesizliği durumunda otomatik dengeleme
x_train, y_train = data_ops.auto_balancer(x_train, y_train)

# Verilerin standardize edilmesi
scaler_params, x_train, x_test = data_ops.standard_scaler(x_train, x_test)

# Configuring optimizator
genetic_optimizer = lambda *args, **kwargs: ene.evolver(*args, policy='aggressive', activation_selection_add_prob=0, activation_mutate_add_prob=0, **kwargs)

# Training Process
model = nn.learn(x_train, y_train, genetic_optimizer, fit_start=True, gen=60)

activation_potentiation = model[model_ops.get_act()]
W = model[model_ops.get_weights()]

# Modeli test etme
test_model = nn.evaluate(x_test, y_test,  W=W, activations=activation_potentiation)
train_model = nn.evaluate(x_train, y_train, W=W, activations=activation_potentiation)

print(classification_report(data_ops.decode_one_hot(y_test), data_ops.decode_one_hot(test_model[model_ops.get_preds()])))
test_acc = test_model[model_ops.get_acc()]
train_acc = train_model[model_ops.get_acc()]

print(Fore.GREEN + "------PLAN Modeli Sonuçları------" + Fore.RESET)
print(f"PLAN Test Accuracy: {test_acc:.4f}")
print(f"PLAN Train Accuracy: {train_acc:.4f}")

model_ops.save_model(model_name='breast_cancer_coimbra',
                scaler_params=scaler_params,
                activation_potentiation=activation_potentiation,
                W=W)

# Model tahminlerini değerlendirme
for i in range(len(x_val)):
    Predict = model_ops.predict_from_storage(model_name='breast_cancer_coimbra', model_path='', Input=x_val[i])

    time.sleep(0.5)
    if np.argmax(Predict) == np.argmax(y_val[i]):
        print(Fore.GREEN + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', np.argmax(y_val[i]))
    else:
        print(Fore.RED + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', np.argmax(y_val[i]))