# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 03:55:15 2024
@author: hasan can beydili

TensorFlow versiyonu - veri işleme PyerualJetwork, model TensorFlow
"""

from colorama import Fore
import numpy as np
from pyerualjetwork.cpu import data_ops
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

import tensorflow as tf
from tensorflow.keras import layers, models

# ──────────────────────────────────────────────
# 1. VERİ YÜKLEME & VEKTÖRIZASYON
# ──────────────────────────────────────────────
newsgroups = fetch_20newsgroups(subset='all')
X = newsgroups.data
y = newsgroups.target

vectorizer = TfidfVectorizer(stop_words='english', max_features=18500)
X = vectorizer.fit_transform(X)
X = X.toarray()

with open('tfidf_20news.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# ──────────────────────────────────────────────
# 2. VERİ AYIRMA & ÖN İŞLEME (PyerualJetwork)
# ──────────────────────────────────────────────
x_train, x_test, y_train, y_test = data_ops.split(X, y, test_size=0.2, random_state=42)
y_train_oh, y_test_oh = data_ops.encode_one_hot(y_train, y_test)
x_train, y_train_oh = data_ops.synthetic_augmentation(x_train, y_train_oh)
scaler_params, x_train, x_test = data_ops.standard_scaler(x_train, x_test)

print('size of training set: %s' % (len(x_train)))
print('size of validation set: %s' % (len(x_test)))

num_classes = y_train_oh.shape[1]

# ──────────────────────────────────────────────
# 3. MODEL (TensorFlow/Keras)
# ──────────────────────────────────────────────
model = models.Sequential([
    layers.Input(shape=(x_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

model.summary()

history = model.fit(
    x_train, y_train_oh,
    epochs=50,
    batch_size=32,
    verbose=1,
)

# ──────────────────────────────────────────────
# 4. DEĞERLENDİRME
# ──────────────────────────────────────────────
test_loss, test_acc = model.evaluate(x_test, y_test_oh, verbose=0)

y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test_oh, axis=1)

print(Fore.GREEN + "\n------TensorFlow Modeli Sonuçları------" + Fore.RESET)
print(f"TF Test Accuracy: {test_acc:.4f}")
print(classification_report(y_true, y_pred, target_names=newsgroups.target_names))