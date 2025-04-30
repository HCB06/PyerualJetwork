from pyerualjetwork.cpu import data_ops, metrics
from pyerualjetwork import nn, ene, model_ops
import numpy as np
import pandas as pd

df = pd.read_csv('MBA.csv') # dataset link: https://www.kaggle.com/datasets/taweilo/mba-admission-dataset/data

y = df['international']

X = df.drop(columns=['international'], axis=1)


# Kategorik sütunları seçme
categorical_columns = X.select_dtypes(include=['object']).columns

# One-Hot Encoding
X = pd.get_dummies(X, columns=categorical_columns)


# Bilinmeyen değerleri "?" olan yerleri NaN ile değiştirme
X.replace('?', np.nan, inplace=True)


X.dropna(inplace=True)

X = np.array(X)

x_train, x_test, y_train, y_test = data_ops.split(X, y, 0.4, 42)
y_train, y_test = data_ops.encode_one_hot(y_train, y_test)

x_train, y_train = data_ops.synthetic_augmentation(x_train, y_train)
x_test, y_test = data_ops.auto_balancer(x_test, y_test)

scaler_params, x_train, x_test= data_ops.standard_scaler(x_train, x_test)

# Configuring optimizer
optimizer = lambda *args, **kwargs: ene.evolver(*args, activation_selection_add_prob=0.85, show_info=True, **kwargs)

model = nn.learn(x_train, y_train, optimizer, model_ops.get_model_template(), fit_start=True, target_acc=1, neurons_history=True)

test_results = nn.evaluate(x_test, y_test, model=model, show_report=True)
test_preds = test_results[model_ops.get_preds()]
test_acc = test_results[model_ops.get_acc()]


precisison, recall, f1 = metrics.metrics(y_test, data_ops.decode_one_hot(test_preds))
print('Precision: ', precisison, '\n', 'Recall: ', recall, '\n', 'F1: ', f1)

# Scaler parametrelerini modele yaz ve artık model kaydedilmeye hazır.
model = model._replace(scaler_params=scaler_params)