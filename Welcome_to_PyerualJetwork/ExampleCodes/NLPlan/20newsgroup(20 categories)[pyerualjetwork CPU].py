# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 03:55:15 2024

@author: hasan can beydili
"""


from colorama import Fore
from pyerualjetwork import neu, data_operations, model_operations, data_operations
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

newsgroups = fetch_20newsgroups(subset='all')

X = newsgroups.data
y = newsgroups.target

vectorizer = TfidfVectorizer(stop_words='english', max_features=18500)

X = vectorizer.fit_transform(X)
X = X.toarray()

with open('tfidf_20news.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

x_train, x_test, y_train, y_test = data_operations.split(X, y, test_size=0.2, random_state=42)
y_train, y_test = data_operations.encode_one_hot(y_train, y_test)
x_train, y_train = data_operations.synthetic_augmentation(x_train, y_train)
scaler_params, x_train, x_test = data_operations.standard_scaler(x_train, x_test)

print('size of training set: %s' % (len(x_train)))
print('size of validation set: %s' % (len(x_test)))
print('classes: %s' % (newsgroups.target_names))

W = neu.fit(x_train, y_train, auto_normalization=False)
model = neu.evaluate(x_test, y_test, W=W)

test_acc = model[model_operations.get_acc()]
test_preds = model[model_operations.get_preds()]
W = model[model_operations.get_weights()]
activation_potentiation = model[model_operations.get_act()]

print(Fore.GREEN + "\n------PLAN Modeli Sonuçları------" + Fore.RESET)
print(f"PLAN Test Accuracy: {test_acc:.4f}")
print(classification_report(data_operations.decode_one_hot(y_test), model[model_operations.get_preds()], target_names=newsgroups.target_names))
