from pyerualjetwork import memory_ops, ene, nn, model_ops
from pyerualjetwork.cpu import  data_ops
data_ops_cpu = data_ops
from pyerualjetwork.cuda import data_ops, metrics

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from termcolor import colored
import numpy as np
import time
import pickle
import cupy as cp

# Veri yükleme ve işleme
data = pd.read_csv('IMDB Dataset.csv') # dataset link: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

X = data['review']
y = data['sentiment']

# Cümlelerin orijinal hallerini kopyalamak için ön ayırma işlemi
x_train, x_test, y_train, y_test = data_ops_cpu.split(X, y, test_size=0.4, random_state=42)

x_test_copy = np.copy(x_test)

# TF-IDF vektörlemesi
vectorizer = TfidfVectorizer(max_features=6000, stop_words='english')
X = vectorizer.fit_transform(X)

# Vectorizer'ı kaydetme
with open('tfidf_imdb.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

X = X.toarray()

# Veriyi eğitim ve test olarak ayırma
x_train, x_test, y_train, y_test = data_ops_cpu.split(X, y, test_size=0.4, random_state=42)

# One-hot encoding işlemi
y_train, y_test = data_ops_cpu.encode_one_hot(y_train, y_test)

# Veri dengeleme işlemi
x_train, y_train = data_ops_cpu.synthetic_augmentation(x_train, y_train)
x_test, y_test = data_ops_cpu.auto_balancer(x_test, y_test)

template_model = model_ops.get_model_template()
scaler_params, x_train, x_test = data_ops.standard_scaler(x_train, x_test)
template_model = template_model._replace(scaler_params=scaler_params)

x_test = memory_ops.transfer_to_gpu(x_test)
y_test = memory_ops.transfer_to_gpu(y_test, dtype=cp.uint8)

# Configuring optimizer
genetic_optimizer = lambda *args, **kwargs: ene.evolver(*args, activation_mutate_add_prob=0, activation_selection_add_prob=0, **kwargs)
model = nn.learn(x_train, y_train, genetic_optimizer, template_model, acc_impact=0, fit_start=True, gen=300, auto_normalization=False)

test_results = nn.evaluate(x_test, y_test, model, cuda=True)

# Test sonuçları ve tahminler
test_acc = test_results[model_ops.get_acc()]
test_preds = test_results[model_ops.get_preds()]

# Modeli kaydetme
model_ops.save_model(model, model_name='IMDB')

# Performans metrikleri
precision, recall, f1 = metrics.metrics(y_test, data_ops.decode_one_hot(test_preds))

print('Precision: ', precision, '\n', 'Recall: ', recall, '\n', 'F1: ', f1)

y_test = data_ops.decode_one_hot(y_test)
test_preds = data_ops.decode_one_hot(test_preds)

# Test verisi üzerinde tahminleri yazdırma
for i in range(len(x_test)):
    
    true_label_text = "positive" if y_test[i] == 1 else "negative"
    pred_text = "positive" if test_preds[i] == 1 else "negative"

    time.sleep(1)

    # Tahminin doğru olup olmadığını kontrol etme
    if y_test[i] == test_preds[i]:
        output = colored(f"Review: {x_test_copy[i]}\nPrediction: {pred_text}\nTrue Label: {true_label_text}\n", 'green')
    else:
        output = colored(f"Review: {x_test_copy[i]}\nPrediction: {pred_text}\nTrue Label: {true_label_text}\n", 'red')

    print(output)
