import pandas as pd
import numpy as np
from pyerualjetwork.cpu import data_ops
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# dataset link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
data = pd.read_csv('creditcard.csv')

x = data.drop('Class', axis=1).values
y = data['Class'].values

x = data_ops.normalization(x)

x_train, x_test, y_train, y_test = data_ops.split(x, y, 0.4, 42)

y_train, y_test = data_ops.encode_one_hot(y_train, y_test)

x_train, y_train = data_ops.synthetic_augmentation(x_train, y_train)

num_classes = y_train.shape[1]
input_size  = x_train.shape[1]

print(f"Eğitim seti : {x_train.shape}")
print(f"Test seti   : {x_test.shape}\n")

# ── Model ─────────────────────────────────────────────────────────────────────
model = keras.Sequential([
    layers.Input(shape=(input_size,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ── Eğitim ───────────────────────────────────────────────────────────────────
print("\nEğitim başlıyor...\n")
model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=32,
    verbose=1,
)

# ── Değerlendirme ─────────────────────────────────────────────────────────────
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy : {test_acc:.4f}")
print(f"Test Loss     : {test_loss:.4f}")

test_preds_proba = model.predict(x_test)
test_preds  = np.argmax(test_preds_proba, axis=1)
y_test_true = np.argmax(y_test, axis=1)

precision = precision_score(y_test_true, test_preds, average="weighted", zero_division=0)
recall    = recall_score   (y_test_true, test_preds, average="weighted", zero_division=0)
f1        = f1_score       (y_test_true, test_preds, average="weighted", zero_division=0)
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

model.save("creditcard_tensorflow_model.keras")
print("\nModel 'creditcard_tensorflow_model.keras' olarak kaydedildi.")
