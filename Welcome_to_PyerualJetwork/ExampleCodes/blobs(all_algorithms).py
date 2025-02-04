# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 03:55:15 2024

@author: hasan
"""

import pandas as pd
import numpy as np
from colorama import Fore
from pyerualjetwork import plan, planeat, data_operations, model_operations
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt

fig, ax = plt.subplots(2, 3)  # Create a new figure and axe

# Karar sınırı çizimi
def plot_decision_boundary(x, y, model, feature_indices=[0, 1], h=0.02, model_name='str', ax=None, which_ax1=None, which_ax2=None, W=None, activation_potentiation=None):
    """
    Plot decision boundary by focusing on specific feature indices.
    
    Parameters:
    - x: Input data
    - y: Target labels
    - model: Trained model
    - feature_indices: Indices of the features to plot (default: [0, 1])
    - h: Step size for the mesh grid
    """
    x_min, x_max = x[:, feature_indices[0]].min() - 1, x[:, feature_indices[0]].max() + 1
    y_min, y_max = x[:, feature_indices[1]].min() - 1, x[:, feature_indices[1]].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Create a full grid with zeros for non-selected features
    grid_full = np.zeros((grid.shape[0], x.shape[1]))
    grid_full[:, feature_indices] = grid
    
    if model == 'PLAN':
        
        Z = [None] * len(grid_full)

        for i in range(len(grid_full)):
            Z[i] = np.argmax(model_operations.predict_model_ram(grid_full[i], W=W, activation_potentiation=activation_potentiation))

        Z = np.array(Z)
        Z = Z.reshape(xx.shape)

    else:

        # Predict on the grid
        Z = model.predict(grid_full)

        if model_name == 'Deep Learning':
            Z = np.argmax(Z, axis=1)  # Get class predictions

        Z = Z.reshape(xx.shape)

    # Plot decision boundary
    ax[which_ax1, which_ax2].contourf(xx, yy, Z, alpha=0.8)
    ax[which_ax1, which_ax2].scatter(x[:, feature_indices[0]], x[:, feature_indices[1]], c=np.argmax(y, axis=1), edgecolors='k', marker='o', s=20, alpha=0.9)
    ax[which_ax1, which_ax2].set_xlabel(f'Feature {feature_indices[0] + 1}')
    ax[which_ax1, which_ax2].set_ylabel(f'Feature {feature_indices[1] + 1}')
    ax[which_ax1, which_ax2].set_title(model_name)


X, y = make_blobs(n_samples=1000, centers=5, random_state=42)

# Eğitim, test ve doğrulama verilerini ayırma
x_train, x_test, y_train, y_test = data_operations.split(X, y, test_size=0.4, random_state=42) # For less train data use this: (X, y, test_size=0.9, random_state=42)

# One-hot encoding işlemi
y_train, y_test = data_operations.encode_one_hot(y_train, y_test)


# Veri dengesizliği durumunu otomatik dengeleme
x_train, y_train = data_operations.auto_balancer(x_train, y_train)
scaler_params, x_train, x_test = data_operations.standard_scaler(x_train, x_test)

# Lojistik Regresyon Modeli
print(Fore.YELLOW + "------Lojistik Regresyon Sonuçları------" + Fore.RESET)
lr_model = LogisticRegression(max_iter=1000, random_state=42)
y_train_decoded = data_operations.decode_one_hot(y_train)
lr_model.fit(x_train, y_train_decoded)

y_test_decoded = data_operations.decode_one_hot(y_test)
y_pred_lr = lr_model.predict(x_test)
test_acc_lr = accuracy_score(y_test_decoded, y_pred_lr)
print(f"Lojistik Regresyon Test Accuracy: {test_acc_lr:.4f}")
print(classification_report(y_test_decoded, y_pred_lr))
# Karar sınırını görselleştir
plot_decision_boundary(x_test, y_test, lr_model, feature_indices=[0, 1], model_name='Logistic Regression', ax=ax, which_ax1=0, which_ax2=0)

# Random Forest Modeli
print(Fore.CYAN + "------Random Forest Sonuçları------" + Fore.RESET)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train_decoded)
y_pred_rf = rf_model.predict(x_test)
test_acc_rf = accuracy_score(y_test_decoded, y_pred_rf)
print(f"Random Forest Test Accuracy: {test_acc_rf:.4f}")
print(classification_report(y_test_decoded, y_pred_rf))
# Karar sınırını görselleştir
plot_decision_boundary(x_test, y_test, rf_model, feature_indices=[0, 1], model_name='Random Forest', ax=ax, which_ax1=0, which_ax2=1)

# XGBoost Modeli
print(Fore.MAGENTA + "------XGBoost Sonuçları------" + Fore.RESET)
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(x_train, y_train_decoded)
y_pred_xgb = xgb_model.predict(x_test)
test_acc_xgb = accuracy_score(y_test_decoded, y_pred_xgb)
print(f"XGBoost Test Accuracy: {test_acc_xgb:.4f}")
print(classification_report(y_test_decoded, y_pred_xgb))
# Karar sınırını görselleştir
plot_decision_boundary(x_test, y_test, xgb_model, feature_indices=[0, 1], model_name='XGBoost', ax=ax, which_ax1=0, which_ax2=2)

# Derin Öğrenme Modeli (Yapay Sinir Ağı)

input_dim = x_train.shape[1]  # Giriş boyutu


model = Sequential()
model.add(Dense(32, activation='relu', input_dim=input_dim))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Model derlemesi
model.compile(optimizer=RMSprop(), loss='binary_crossentropy', metrics=['accuracy'])

# Model eğitimi (early stopping ile)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
model.fit(x_train, y_train, epochs=50, batch_size=32, callbacks=[early_stop], verbose=1)

# Test verileri üzerinde modelin performansını değerlendirme
y_pred_dl = model.predict(x_test)
y_pred_dl_classes = np.argmax(y_pred_dl, axis=1)  # Tahmin edilen sınıflar
y_test_decoded_dl = data_operations.decode_one_hot(y_test)
print(Fore.BLUE + "------Derin Öğrenme (ANN) Sonuçları------" + Fore.RESET)
test_acc_dl = accuracy_score(y_test_decoded_dl, y_pred_dl_classes)
print(f"Derin Öğrenme Test Accuracy: {test_acc_dl:.4f}")
print(classification_report(y_test_decoded_dl, y_pred_dl_classes))
# Karar sınırını görselleştir
plot_decision_boundary(x_test, y_test, model=model, feature_indices=[0, 1], model_name='Deep Learning', ax=ax, which_ax1=1, which_ax2=0)

# PLAN Modeli
# Configuring optimizer
genetic_optimizer = lambda *args, **kwargs: planeat.evolver(*args, activation_mutate_add_prob=0, activation_selection_add_prob=0, **kwargs)

model = plan.learner(x_train, y_train, genetic_optimizer, loss_impact=0, batch_size=0.1, gen=80)

W = model[model_operations.get_weights()]
activation_potentiation = model[model_operations.get_act_pot()]

test_model = plan.evaluate(x_test, y_test, W=W, activation_potentiation=activation_potentiation)
test_acc_plan = test_model[model_operations.get_acc()]
print(Fore.GREEN + "------PLAN Modeli Sonuçları------" + Fore.RESET)
print(f"PLAN Test Accuracy: {test_acc_plan:.4f}")
print(classification_report(data_operations.decode_one_hot(y_test), data_operations.decode_one_hot(test_model[model_operations.get_preds_softmax()])))
# Karar sınırını görselleştir
plot_decision_boundary(x_test, y_test, model='PLAN', feature_indices=[0, 1], model_name='PLAN', ax=ax, which_ax1=1, which_ax2=1, W=W, activation_potentiation=activation_potentiation)
plt.show()
