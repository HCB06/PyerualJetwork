# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 03:55:15 2024

@author: hasan can beydili
"""

from colorama import Fore
from pyerualjetwork import neu, data_operations, model_operations
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

newsgroup = fetch_20newsgroups(subset='all', categories=categories)
X = newsgroup.data
y = newsgroup.target

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

X = vectorizer.fit_transform(X)
X = X.toarray()

# Eğitim ve test verilerine ayrıma
x_train, x_test, y_train, y_test = data_operations.split(X, y, test_size=0.4, random_state=42)


# One-hot encoding işlemi
y_train, y_test = data_operations.encode_one_hot(y_train, y_test)


# Veri dengesizliği durumunu otomatik dengeleme
x_train, y_train = data_operations.synthetic_augmentation(x_train, y_train)

scaler_params, x_train, x_test = data_operations.standard_scaler(x_train, x_test)

print('size of training set: %s' % (len(x_train)))
print('size of validation set: %s' % (len(x_test)))
print('classes: %s' % (newsgroup.target_names))


# PLAN Modeli
genetic_optimizer = lambda *args, **kwargs: ene.evolver(*args, **kwargs)
model = neu.learner(x_train, y_train, genetic_optimizer, fit_start=True, gen=2, neurons_history=True, target_acc=1)

W = model[model_operations.get_weights()]
activation_potentiation = model[model_operations.get_act_pot()]

test_model = neu.evaluate(x_test, y_test, W=W, show_metrics=True, activations=activation_potentiation)

# Modeli test etme
test_acc_plan = test_model[model_operations.get_acc()]
print(Fore.GREEN + "\n------PLAN Modeli Sonuçları------" + Fore.RESET)
print(f"PLAN Test Accuracy: {test_acc_plan:.4f}")
print(classification_report(data_operations.decode_one_hot(y_test), test_model[model_operations.get_preds()], target_names=newsgroup.target_names))
