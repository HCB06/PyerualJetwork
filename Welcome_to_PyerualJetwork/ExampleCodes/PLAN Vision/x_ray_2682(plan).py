import time
from colorama import Fore, Style
from pyerualjetwork import neu_cpu
from pyerualjetwork import data_operations_cpu, model_operations_cpu
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
import os

# dataset link: https://mega.nz/file/TcIyWSzT#CpOImZ5OjwKJjP6KQpkEnFka6TwdRcEKvaz7LwgBEww

def load_and_preprocess_images(folder_path, target_size=(1200, 1200)):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, target_size)
                images.append(img_resized)
    return images

folder_path = 'chest_xray/train/NORMAL'
x_train_0 = load_and_preprocess_images(folder_path)

folder_path = 'chest_xray/train/PNEUMONIA'
x_train_1 = load_and_preprocess_images(folder_path)

x_train = x_train_0 + x_train_1

folder_path = 'chest_xray/test/NORMAL'
x_test_0 = load_and_preprocess_images(folder_path)

folder_path = 'chest_xray/test/PNEUMONIA'
x_test_1 = load_and_preprocess_images(folder_path)

x_test = x_test_0 + x_test_1


y_train_0 = [0] * len(x_train_0)
y_train_1 = [1] * len(x_train_1)


y_test_0 = [0] * len(x_test_0)
y_test_1 = [1] * len(x_test_1)

y_train = y_train_0 + y_train_1
y_test = y_test_0 + y_test_1

y_train = np.array(y_train)
y_test = np.array(y_test)

y_train, y_test = data_operations_cpu.encode_one_hot(y_train, y_test)


x_train, y_train = data_operations_cpu.auto_balancer(x_train, y_train)
x_test, y_test = data_operations_cpu.auto_balancer(x_test, y_test)


x_train = data_operations_cpu.normalization(x_train)
x_test = data_operations_cpu.normalization(x_test)


activation_potentiation = ['bent_identity']

W = neu_cpu.fit(x_train, y_train, activations=activation_potentiation)

test_model = neu_cpu.evaluate(x_test, y_test, show_metrices=True, W=W, activations=activation_potentiation)

test_preds = test_model[model_operations_cpu.get_preds()]
test_acc = test_model[model_operations_cpu.get_acc()]
# Test verisinden rastgele 12 örnek seç
num_samples = 12
random_indices = random.sample(range(len(x_test)), num_samples)

# Seçilen test örneklerinin tahminlerini yap ve ekranda göster
fig, axs = plt.subplots(3, 4, figsize=(12, 9))
axs = axs.ravel()  # Axes'i düzleştir


for i, idx in enumerate(random_indices):
    img = x_test[idx]  # Resim
    true_label = np.argmax(y_test[idx])  # Gerçek etiket
    pred_label = test_preds[idx]  # Tahmin edilen etiket
    
    # Resmi göster
    axs[i].imshow(img, cmap='gray')
    
    # Gerçek ve tahmin edilen etiketi göster
    if pred_label == 0:
        label = 'Normal'
    else:
        label = 'Hasta'
    
    # Tahmin doğruysa başlık yeşil, yanlışsa kırmızı
    if true_label == pred_label:
        title_color = 'green'  # Doğru tahmin
    else:
        title_color = 'red'  # Yanlış tahmin
    
    # Resmin başlığına tahmini ekle
    axs[i].set_title(f'Tahmin: {label}', color=title_color)
    axs[i].axis('off')  # Eksenleri kapat

# Son düzenlemeleri yap ve göster
plt.tight_layout()
plt.show()
