# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 03:55:15 2024

@author: hasan can beydili
"""


from colorama import Fore
from pyerualjetwork.cpu import data_ops
from pyerualjetwork import nn, model_ops, ene
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

x_train, x_test, y_train, y_test = data_ops.split(X, y, test_size=0.2, random_state=42)
y_train, y_test = data_ops.encode_one_hot(y_train, y_test)
x_train, y_train = data_ops.synthetic_augmentation(x_train, y_train)
scaler_params, x_train, x_test = data_ops.standard_scaler(x_train, x_test)

print('size of training set: %s' % (len(x_train)))
print('size of validation set: %s' % (len(x_test)))

genetic_optimizer = lambda *args, **kwargs: ene.evolver(*args, **kwargs)
gradient_optimizer = lambda *args, **kwargs: nn.grad(*args, method='adam', learning_rate=0.01, **kwargs)


#--- PLAN ----
print("PLAN training starting...")

model = nn.learn(x_train, 
                 y_train, 
                 iter=1, 
                 backprop_train=False,
                 quick_start=False,
                 fit_start=True, 
                 genetic_optimizer=genetic_optimizer, 
                 pop_size=30)

test_results = nn.evaluate(x_test, y_test, model=model)

test_acc = test_results[model_ops.get_acc()]

print(Fore.GREEN + "\n------PLAN Modeli Sonuçları------" + Fore.RESET)
print(f"PLAN Test Accuracy: {test_acc:.4f}")
print(classification_report(data_ops.decode_one_hot(y_test), data_ops.decode_one_hot(test_results[model_ops.get_preds()]), target_names=newsgroups.target_names))


#--- PTNN ----
print("PTNN training starting...")

model = nn.learn(x_train, 
                 y_train, 
                 iter=[1,5], 
                 neurons=[64], 
                 activation_functions=['relu'], 
                 backprop_train=True, 
                 step_size=32, 
                 quick_start=True, 
                 fit_start=True, 
                 genetic_optimizer=genetic_optimizer, 
                 gradient_optimizer=gradient_optimizer, 
                 pop_size=30)

test_results = nn.evaluate(x_test, y_test, model=model)

test_acc = test_results[model_ops.get_acc()]

print(Fore.GREEN + "\n------PTNN Modeli Sonuçları------" + Fore.RESET)
print(f"PTNN Test Accuracy: {test_acc:.4f}")
print(classification_report(data_ops.decode_one_hot(y_test), data_ops.decode_one_hot(test_results[model_ops.get_preds()]), target_names=newsgroups.target_names))


#--- MLP ----
print("MLP training starting...")

model = nn.learn(x_train, 
                 y_train, 
                 iter=20, 
                 neurons=[64], 
                 activation_functions=['relu'], 
                 backprop_train=True, 
                 step_size=32, 
                 quick_start=False, 
                 fit_start=False, 
                 genetic_optimizer=genetic_optimizer, 
                 gradient_optimizer=gradient_optimizer)

test_results = nn.evaluate(x_test, y_test, model=model)

test_acc = test_results[model_ops.get_acc()]

print(Fore.GREEN + "\n------MLP Modeli Sonuçları------" + Fore.RESET)
print(f"MLP Test Accuracy: {test_acc:.4f}")
print(classification_report(data_ops.decode_one_hot(y_test), data_ops.decode_one_hot(test_results[model_ops.get_preds()]), target_names=newsgroups.target_names))

