from pyerualjetwork import neu
from pyerualjetwork import data_operations, model_operations
import numpy as np
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

folder_path = 'chest_xray/val/NORMAL'
x_train_0 = load_and_preprocess_images(folder_path)

folder_path = 'chest_xray/val/PNEUMONIA'
x_train_1 = load_and_preprocess_images(folder_path)


folder_path = 'chest_xray/test/NORMAL'
x_test_0 = load_and_preprocess_images(folder_path)

folder_path = 'chest_xray/test/PNEUMONIA'
x_test_1 = load_and_preprocess_images(folder_path)

x_test = x_test_0 + x_test_1

x_train = x_train_0 + x_train_1


y_train_0 = [0] * len(x_train_0)
y_train_1 = [1] * len(x_train_1)

y_test_0 = [0] * len(x_test_0)
y_test_1 = [1] * len(x_test_1)

y_train = y_train_0 + y_train_1
y_test = y_test_0 + y_test_1

y_train = np.array(y_train)
y_test = np.array(y_test)

y_train, y_test = data_operations.encode_one_hot(y_train, y_test)

x_train, y_train = data_operations.auto_balancer(x_train, y_train)
x_test, y_test = data_operations.auto_balancer(x_test, y_test)

x_train = data_operations.normalization(x_train)
x_test = data_operations.normalization(x_test)

W = neu.plan_fit(x_train, y_train, auto_normalization=False)

test_model = neu.evaluate(x_test, y_test, W=W)

output = [0,1]
Input = model_operations.reverse_predict_from_memory(output, W)

plt.imshow(np.reshape(Input, (1200,1200)))
plt.show()