from pyerualjetwork import neu, ene, data_operations, model_operations, metrics
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

x_train, x_test, y_train, y_test = data_operations.split(X, y, 0.4, 42)
y_train, y_test = data_operations.encode_one_hot(y_train, y_test)

x_train, y_train = data_operations.synthetic_augmentation(x_train, y_train)
x_test, y_test = data_operations.auto_balancer(x_test, y_test)

scaler_params, x_train, x_test= data_operations.standard_scaler(x_train, x_test)

# Configuring optimizer
optimizer = lambda *args, **kwargs: ene.evolver(*args, activation_selection_add_prob=0.85, show_info=True, **kwargs)

model = neu.learn(x_train, y_train, optimizer, fit_start=True, target_acc=1, neurons_history=True)

test_model = neu.evaluate(x_test, y_test, W=model[model_operations.get_weights()], activations=model[model_operations.get_act()])
test_preds = test_model[model_operations.get_preds()]
test_acc = test_model[model_operations.get_acc()]


precisison, recall, f1 = metrics.metrics(y_test, data_operations.decode_one_hot(test_preds))
print('Precision: ', precisison, '\n', 'Recall: ', recall, '\n', 'F1: ', f1)
