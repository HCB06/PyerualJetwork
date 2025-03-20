import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from pyerualjetwork import data_operations
import numpy as np
from matplotlib import pyplot as plt

# Veriyi yükleme
file_path = 'breast_cancer_coimbra.csv'
data = pd.read_csv(file_path)


X = data.drop('Classification', axis=1).values
y = data['Classification'].values

# Eğitim, test ve doğrulama verilerini ayırma
x_train, x_test, y_train, y_test = data_operations.split(X, y, 0.4, 42)
x_train, x_val, y_train, y_val = data_operations.split(x_train, y_train, 0.2, 42)

y_val = data_operations.encode_one_hot(y_val, y_test)[0]
y_train, y_test = data_operations.encode_one_hot(y_train, y_test)

# Veri dengesizliği durumunda otomatik dengeleme
x_train, y_train = data_operations.auto_balancer(x_train, y_train)

y_train = data_operations.decode_one_hot(y_train)
y_test = data_operations.decode_one_hot(y_test)

# Verilerin standardize edilmesi
scaler_params, x_train, x_test = data_operations.standard_scaler(x_train, x_test)

# Lojistik Regresyon modelini tanımlama ve eğitme
model = LogisticRegression(random_state=42)
model.fit(x_train, y_train)

# Modeli test etme
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
test_acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_acc:.4f}")

x_val = data_operations.standard_scaler(x_test=x_val, scaler_params=scaler_params)

# Model tahminlerini değerlendirme
y_val_pred = model.predict(x_val)
for i in range(len(x_val)):
    print(f"Predicted Output: {y_val_pred[i]}, Real Output: {y_val[i]}")


# Karar sınırlarını görselleştirme
feature_indices = [0, 1]

h = .02  # mesh grid adımı
x_min, x_max = x_test[:, feature_indices[0]].min() - 1, x_test[:, feature_indices[0]].max() + 1
y_min, y_max = x_test[:, feature_indices[1]].min() - 1, x_test[:, feature_indices[1]].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

grid = np.c_[xx.ravel(), yy.ravel()]
grid_full = np.zeros((grid.shape[0], x_test.shape[1]))
grid_full[:, feature_indices] = grid

Z = model.predict(grid_full)

Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(x_test[:, feature_indices[0]], x_test[:, feature_indices[1]], c=y_test, edgecolors='k', marker='o', s=20, alpha=0.9)
plt.xlabel(f'Feature {feature_indices[0] + 1}')
plt.ylabel(f'Feature {feature_indices[1] + 1}')
plt.title('Decision Boundary')
plt.show()
