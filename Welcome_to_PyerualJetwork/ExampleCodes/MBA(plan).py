from pyerualjetwork import neu_cpu, ene_cpu, data_operations_cpu, model_operations_cpu, metrics_cpu
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

x_train, x_test, y_train, y_test = data_operations_cpu.split(X, y, 0.4, 42)
y_train, y_test = data_operations_cpu.encode_one_hot(y_train, y_test)

x_train, y_train = data_operations_cpu.synthetic_augmentation(x_train, y_train)
x_test, y_test = data_operations_cpu.auto_balancer(x_test, y_test)

scaler_params, x_train, x_test= data_operations_cpu.standard_scaler(x_train, x_test)

# Configuring optimizer
optimizer = lambda *args, **kwargs: ene_cpu.evolver(*args, activation_selection_add_prob=0.85, show_info=True, **kwargs)

model = neu_cpu.learn(x_train, y_train, optimizer, fit_start=True, target_acc=1, neurons_history=True)

test_model = neu_cpu.evaluate(x_test, y_test, W=model[model_operations_cpu.get_weights()], activations=model[model_operations_cpu.get_act()])
test_preds = test_model[model_operations_cpu.get_preds()]
test_acc = test_model[model_operations_cpu.get_acc()]


precisison, recall, f1 = metrics_cpu.metrics_cpu(y_test, data_operations_cpu.decode_one_hot(test_preds))
print('Precision: ', precisison, '\n', 'Recall: ', recall, '\n', 'F1: ', f1)
