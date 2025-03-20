import pandas as pd
import numpy as np
from colorama import Fore
from pyerualjetwork import neu, ene, data_operations, model_operations
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv('kag_risk_factors_cervical_cancer.csv') # dataset link: https://www.kaggle.com/datasets/loveall/cervical-cancer-risk-classification


# `?` karakterlerini NaN ile değiştirme
df.replace('?', np.nan, inplace=True)

# Eksik değerleri sütunların ortalama değerleri ile doldurma
df = df.astype(float)  # Tüm sütunları sayısal veri türüne dönüştürme
df.fillna(df.mean(), inplace=True)

# Son sütunu y olarak ayır
y = df.iloc[:, -1]  # Son sütun Y olacak

# Diğer sütunlar X olacak
X = df.iloc[:, :-1]  # Son sütun hariç diğer sütunlar X olacak

# Numpy dizisine çevirme
X = np.array(X)
y = np.array(y)

# Veriyi eğitim ve test setlerine ayırma
x_train, x_test, y_train, y_test = data_operations.split(X, y, 0.4, 42)

# One-hot encoding
y_train, y_test = data_operations.encode_one_hot(y_train, y_test)

# Veri dengeleme
x_train, y_train = data_operations.synthetic_augmentation(x_train, y_train)
x_test, y_test = data_operations.auto_balancer(x_test, y_test)

# Ölçekleme
scaler_params, x_train, x_test = data_operations.standard_scaler(x_train, x_test)


# Lojistik Regresyon Modeli
print(Fore.YELLOW + "------Lojistik Regresyon Sonuçları------" + Fore.RESET)
lr_model = LogisticRegression(max_iter=1000, random_state=42)
y_train_decoded = data_operations.decode_one_hot(y_train)  # One-hot encoded y_train verilerini geri dönüştürme
lr_model.fit(x_train, y_train_decoded)  # Modeli eğitme
y_test_decoded = data_operations.decode_one_hot(y_test)
y_pred_lr = lr_model.predict(x_test)
test_acc_lr = accuracy_score(y_test_decoded, y_pred_lr)
print(f"Lojistik Regresyon Test Accuracy: {test_acc_lr:.4f}")
print(classification_report(y_test_decoded, y_pred_lr))

# Random Forest Modeli
print(Fore.CYAN + "------Random Forest Sonuçları------" + Fore.RESET)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train_decoded)  # Modeli eğitme
y_pred_rf = rf_model.predict(x_test)
test_acc_rf = accuracy_score(y_test_decoded, y_pred_rf)
print(f"Random Forest Test Accuracy: {test_acc_rf:.4f}")
print(classification_report(y_test_decoded, y_pred_rf))

# XGBoost Modeli
print(Fore.MAGENTA + "------XGBoost Sonuçları------" + Fore.RESET)
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(x_train, y_train_decoded)  # Modeli eğitme
y_pred_xgb = xgb_model.predict(x_test)
test_acc_xgb = accuracy_score(y_test_decoded, y_pred_xgb)
print(f"XGBoost Test Accuracy: {test_acc_xgb:.4f}")
print(classification_report(y_test_decoded, y_pred_xgb))

# Derin Öğrenme Modeli (Yapay Sinir Ağı)
input_dim = x_train.shape[1]  # Giriş boyutu

model = Sequential()
model.add(Dense(16, input_dim=input_dim, activation='tanh'))  # Giriş katmanı ve ilk gizli katman
model.add(Dense(y_train.shape[1], activation='softmax'))  # Çıkış katmanı (softmax)

# Modeli derleme
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test), verbose=2)

# Test verileri üzerinde modelin performansını değerlendirme
y_pred_dl = model.predict(x_test)
y_pred_dl_classes = np.argmax(y_pred_dl, axis=1)  # Tahmin edilen sınıflar
y_test_decoded_dl = data_operations.decode_one_hot(y_test)

print(Fore.BLUE + "------Derin Öğrenme (ANN) Sonuçları------" + Fore.RESET)
test_acc_dl = accuracy_score(y_test_decoded_dl, y_pred_dl_classes)
print(f"Derin Öğrenme Test Accuracy: {test_acc_dl:.4f}")
print(classification_report(y_test_decoded_dl, y_pred_dl_classes))

# PLAN Modeli
# Configuring optimizer
genetic_optimizer = lambda *args, **kwargs: ene.evolver(*args, activation_mutate_add_prob=0, activation_selection_add_prob=0, policy='aggressive', **kwargs)

model = neu.learn(x_train, y_train, genetic_optimizer, gen=17, batch_size=0.05)

test_model = neu.evaluate(x_test, y_test, W=model[model_operations.get_weights()], activations=model[model_operations.get_act()])
train_model = neu.evaluate(x_train, y_train, W=model[model_operations.get_weights()], activations=model[model_operations.get_act()])

print(Fore.GREEN + "\n------PLAN Modeli Sonuçları------" + Fore.RESET)
test_acc_plan = test_model[model_operations.get_acc()]
train_acc_plan = train_model[model_operations.get_acc()]

print(f"PLAN Test Accuracy: {test_acc_plan:.4f}")
print(f"PLAN Train Accuracy: {train_acc_plan:.4f}")
print(classification_report(data_operations.decode_one_hot(y_test), data_operations.decode_one_hot(test_model[model_operations.get_preds()])))
