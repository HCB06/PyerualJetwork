import numpy as np
import tensorflow as tf
import pandas as pd
from pyerualjetwork import data_operations
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Breast Cancer Coimbra veri setini yükleme (manuel indirilmiş)
file_path = 'breast_cancer_coimbra.csv'  # Bu dosya yolunu indirdiğiniz dosyaya göre değiştirin
data = pd.read_csv(file_path)

# Özellikler ve etiketleri ayırma (Bu, veri setinin yapısına bağlı olarak değişebilir)
# 'Class' sütununu etiket olarak varsayıyoruz, bu sütunun adını veri setinize göre değiştirmelisiniz
X = data.drop('Classification', axis=1).values  # 'Class' sütunu etiket olarak alınır
y = data['Classification'].values

# Eğitim ve test verilerini ayırma
x_train, x_test, y_train, y_test = data_operations.split(X, y, 0.4, 42)
x_train, x_val, y_train, y_val = data_operations.split(x_train, y_train, 0.2, 42)


# One-hot encoding işlemi
y_val = data_operations.encode_one_hot(y_val, y_test)[0]
y_train, y_test = data_operations.encode_one_hot(y_train, y_test)

# Veri dengesizliği durumunda otomatik dengeleme
x_train, y_train = data_operations.auto_balancer(x_train, y_train)

# Verilerin standardize edilmesi
scaler_params, x_train, x_test = data_operations.standard_scaler(x_train, x_test)

# Erken durdurma callback'i
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='train_loss', 
    patience=10,  # Eğitim 10 epoch boyunca iyileşme göstermiyorsa durur
    restore_best_weights=True  # Modelin en iyi halini geri yükler
)

# Dropout ve L2 düzenlileştirme ile derin öğrenme modeli
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, input_dim=x_train.shape[1], activation='relu' ),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dense(2, activation='softmax')
])


# Modeli derleme
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
history = model.fit(x_train, y_train, epochs=15, batch_size=32)

# Modeli test etme
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Doğruluk ve rapor hesaplama
accuracy = accuracy_score(y_test_classes, y_pred_classes)
print(f"Test Doğruluğu: {accuracy * 100:.2f}%")
print("Sınıflandırma Raporu:")
print(classification_report(y_test_classes, y_pred_classes))

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
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(x_test[:, feature_indices[0]], x_test[:, feature_indices[1]], c=y_test_classes, edgecolors='k', marker='o', s=20, alpha=0.9)
plt.xlabel(f'Feature {feature_indices[0] + 1}')
plt.ylabel(f'Feature {feature_indices[1] + 1}')
plt.title('Decision Boundary')
plt.show()
