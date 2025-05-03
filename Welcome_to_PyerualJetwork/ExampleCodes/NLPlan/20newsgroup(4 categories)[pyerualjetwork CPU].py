# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 03:55:15 2024

@author: hasan can beydili
"""

from colorama import Fore
from pyerualjetwork.cpu import data_ops
from pyerualjetwork import nn, ene, model_ops
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

newsgroup = fetch_20newsgroups(subset='all', categories=categories)
X = newsgroup.data
y = newsgroup.target

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

X = vectorizer.fit_transform(X)
X = X.toarray()

# Eğitim ve test verilerine ayrıma
x_train, x_test, y_train, y_test = data_ops.split(X, y, test_size=0.4, random_state=42)


# One-hot encoding işlemi
y_train, y_test = data_ops.encode_one_hot(y_train, y_test)

# Veri dengesizliği durumunu otomatik dengeleme
x_train, y_train = data_ops.synthetic_augmentation(x_train, y_train)

scaler_params, x_train, x_test = data_ops.standard_scaler(x_train, x_test)

print('size of training set: %s' % (len(x_train)))
print('size of validation set: %s' % (len(x_test)))
print('classes: %s' % (newsgroup.target_names))


# PLAN Modeli
genetic_optimizer = lambda *args, **kwargs: ene.evolver(*args, **kwargs)
model = nn.learn(x_train, y_train, genetic_optimizer, fit_start=True, gen=2, neurons_history=True, target_acc=1, pop_size=100)

test_model = nn.evaluate(x_test, y_test, show_report=True, model=model)

# Scaler parametrelerini modele yaz
model = model._replace(scaler_params=scaler_params)

# Modeli test etme
test_acc_plan = test_model[model_ops.get_acc()]
print(Fore.GREEN + "\n------PLAN Modeli Sonuçları------" + Fore.RESET)
print(f"PLAN Test Accuracy: {test_acc_plan:.4f}")
print(classification_report(data_ops.decode_one_hot(y_test), test_model[model_ops.get_preds()], target_names=newsgroup.target_names))
