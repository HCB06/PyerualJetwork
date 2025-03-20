# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 05:43:52 2024

@author: hasan
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data
y = data.target



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=688)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

encoder = OneHotEncoder()
TrainLabels = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
TestLabels = encoder.transform(y_test.reshape(-1, 1)).toarray()

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train_scaled, TrainLabels, epochs=30, batch_size=32, validation_data=(X_test_scaled, TestLabels))

test_loss, test_accuracy = model.evaluate(X_test_scaled, TestLabels)
