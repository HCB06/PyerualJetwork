# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 03:55:15 2024

@author: hasan
"""

import pandas as pd
import numpy as np
from pyerualjetwork import neu, ene, data_operations, model_operations
from colorama import Fore
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('survey_lung_cancer.csv') #https://www.kaggle.com/datasets/ajisofyan/survey-lung-cancer
y = df.iloc[:, -1]
X = df.drop(columns=df.columns[-1])


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

categorical_columns = X.select_dtypes(include=['object']).columns
for col in categorical_columns:
    label_encoder = LabelEncoder()
    X[col] = label_encoder.fit_transform(X[col])

X = np.array(X)
x_train, x_test, y_train, y_test = data_operations.split(X, y, test_size=0.4, random_state=42)

y_train, y_test = data_operations.encode_one_hot(y_train, y_test)

x_train, y_train = data_operations.auto_balancer(x_train, y_train)
x_test, y_test = data_operations.auto_balancer(x_test, y_test)

scaler_params, x_train, x_test = data_operations.standard_scaler(x_train, x_test)

# Lojistik Regresyon Modeli
print(Fore.YELLOW + "------Lojistik Regresyon Sonuçları------" + Fore.RESET)
lr_model = LogisticRegression(max_iter=1000, random_state=42)
y_train_decoded = data_operations.decode_one_hot(y_train)
lr_model.fit(x_train, y_train_decoded)

y_test_decoded = data_operations.decode_one_hot(y_test)
y_pred_lr = lr_model.predict(x_train)
train_acc_lr = accuracy_score(y_train_decoded, y_pred_lr)
#print(f"Lojistik Regresyon Train Accuracy: {train_acc_lr:.4f}")

y_pred_lr = lr_model.predict(x_test)
test_acc_lr = accuracy_score(y_test_decoded, y_pred_lr)
print(f"Lojistik Regresyon Test Accuracy: {test_acc_lr:.4f}")
print(classification_report(y_test_decoded, y_pred_lr))


# SVM Modeli
print(Fore.RED + "------SVM Sonuçları------" + Fore.RESET)
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(x_train, y_train_decoded)

y_pred_svm = svm_model.predict(x_train)
train_acc_svm = accuracy_score(y_train_decoded, y_pred_svm)
#print(f"SVM Train Accuracy: {train_acc_svm:.4f}")

y_pred_svm = svm_model.predict(x_test)
test_acc_svm = accuracy_score(y_test_decoded, y_pred_svm)
print(f"SVM Test Accuracy: {test_acc_svm:.4f}")
print(classification_report(y_test_decoded, y_pred_svm))


# Random Forest Modeli
print(Fore.CYAN + "------Random Forest Sonuçları------" + Fore.RESET)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train_decoded)
y_pred_rf = rf_model.predict(x_train)
train_acc_rf = accuracy_score(y_train_decoded, y_pred_rf)
#print(f"Random Forest Train Accuracy: {train_acc_rf:.4f}")

y_pred_rf = rf_model.predict(x_test)
test_acc_rf = accuracy_score(y_test_decoded, y_pred_rf)
print(f"Random Forest Test Accuracy: {test_acc_rf:.4f}")
print(classification_report(y_test_decoded, y_pred_rf))


# XGBoost Modeli
print(Fore.MAGENTA + "------XGBoost Sonuçları------" + Fore.RESET)
xgb_model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
xgb_model.fit(x_train, y_train_decoded)

y_pred_xgb = xgb_model.predict(x_train)
train_acc_xgb = accuracy_score(y_train_decoded, y_pred_xgb)
#print(f"XGBoost Train Accuracy: {train_acc_xgb:.4f}")

y_pred_xgb = xgb_model.predict(x_test)
test_acc_xgb = accuracy_score(y_test_decoded, y_pred_xgb)
print(f"XGBoost Test Accuracy: {test_acc_xgb:.4f}")
print(classification_report(y_test_decoded, y_pred_xgb))

# Derin Öğrenme Modeli (Yapay Sinir Ağı)

input_dim = x_train.shape[1]  # Giriş boyutu

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=input_dim))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Model derlemesi
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model eğitimi (early stopping ile)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
model.fit(x_train, y_train, epochs=60, batch_size=32, callbacks=[early_stop], verbose=1)

# Test verileri üzerinde modelin performansını değerlendirme
print(Fore.BLUE + "------Derin Öğrenme (ANN) Sonuçları------" + Fore.RESET)
y_pred_dl = model.predict(x_train, verbose=0)
y_pred_dl_classes = np.argmax(y_pred_dl, axis=1)  # Tahmin edilen sınıflar
y_train_decoded_dl = data_operations.decode_one_hot(y_train)
train_acc_dl = accuracy_score(y_train_decoded_dl, y_pred_dl_classes)
#print(f"Derin Öğrenme Train Accuracy: {train_acc_dl:.4f}")

y_pred_dl = model.predict(x_test, verbose=0)
y_pred_dl_classes = np.argmax(y_pred_dl, axis=1)  # Tahmin edilen sınıflar
y_test_decoded_dl = data_operations.decode_one_hot(y_test)
test_acc_dl = accuracy_score(y_test_decoded_dl, y_pred_dl_classes)
print(f"Derin Öğrenme Test Accuracy: {test_acc_dl:.4f}")
print(classification_report(y_test_decoded_dl, y_pred_dl_classes))


# PLAN Modeli
# Configuring optimizer
genetic_optimizer = lambda *args, **kwargs: ene.evolver(*args, **kwargs)
model = neu.learn(x_train, y_train, genetic_optimizer, auto_normalization=False,
                     gen=5)

W = model[model_operations.get_weights()]
activation_potentiation = model[model_operations.get_act_pot()]

print(Fore.GREEN + "------PLAN Modeli Sonuçları------" + Fore.RESET)
train_model = neu.evaluate(x_train, y_train, W=W, activations=activation_potentiation)
train_acc_plan = train_model[model_operations.get_acc()]
#print(f"PLAN Train Accuracy: {train_acc_neu:.4f}")

test_model = neu.evaluate(x_test, y_test, W=W, activations=activation_potentiation)
test_acc_plan = test_model[model_operations.get_acc()]
print(f"PLAN Test Accuracy: {test_acc_plan:.4f}")
print(classification_report(data_operations.decode_one_hot(y_test), data_operations.decode_one_hot(test_model[model_operations.get_preds()])))
