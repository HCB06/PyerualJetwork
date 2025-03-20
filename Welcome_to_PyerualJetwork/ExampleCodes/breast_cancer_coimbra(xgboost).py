import pandas as pd
import numpy as np
from colorama import Fore
from pyerualjetwork import data_operations
import time
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

file_path = 'breast_cancer_coimbra.csv' 
data = pd.read_csv(file_path)

X = data.drop('Classification', axis=1).values
y = data['Classification'].values

# Eğitim, test ve doğrulama verilerini ayırma
x_train, x_test, y_train, y_test = data_operations.split(X, y, 0.4, 42)
x_train, x_val, y_train, y_val = data_operations.split(x_train, y_train, 0.2, 42)

# One-hot encoding işlemi
y_train, y_test = data_operations.encode_one_hot(y_train, y_test)
y_val = data_operations.encode_one_hot(y_val, y)[0]

# Veri dengesizliği durumunda otomatik dengeleme
x_train, y_train = data_operations.auto_balancer(x_train, y_train)

# Verilerin standardize edilmesi
scaler_params, x_train, x_test = data_operations.standard_scaler(x_train, x_test)

# XGBoost modelini oluşturma
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Modeli eğitme
xgb_model.fit(x_train, np.argmax(y_train, axis=1))

# Test seti ile tahmin
y_pred = xgb_model.predict(x_test)

# Test doğruluğunu hesaplama
test_acc = accuracy_score(np.argmax(y_test, axis=1), y_pred)
print(f"Test Accuracy: {test_acc}")

# Classification report
print("\nClassification Report:")
print(classification_report(np.argmax(y_test, axis=1), y_pred))

# Model tahminlerini değerlendirme
for i in range(len(x_val)):
    Predict = xgb_model.predict(x_val[i].reshape(1, -1))

    time.sleep(0.5)
    if Predict == np.argmax(y_val[i]):
        print(Fore.GREEN + f'Predicted Output(index): {Predict[0]}, Real Output(index): {np.argmax(y_val[i])}')
    else:
        print(Fore.RED + f'Predicted Output(index): {Predict[0]}, Real Output(index): {np.argmax(y_val[i])}')
