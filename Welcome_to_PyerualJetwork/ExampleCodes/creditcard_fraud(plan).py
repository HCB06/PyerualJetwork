import pandas as pd
from pyerualjetwork.cpu import data_ops
from pyerualjetwork import nn, ene, model_ops

data = pd.read_csv('creditcard.csv') # dataset link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

x = data.drop('Class', axis=1).values
y = data['Class'].values

x = data_ops.normalization(x)
x_train, x_test, y_train, y_test = data_ops.split(x, y, 0.4, 42)

y_train, y_test = data_ops.encode_one_hot(y_train, y_test)

scaler_params, x_train, x_test = data_ops.standard_scaler(x_train, x_test)

x_train, y_train = data_ops.synthetic_augmentation(x_train, y_train)

# Configuring optimizer
genetic_optimizer = lambda *args, **kwargs: ene.evolver(*args, **kwargs)
model = nn.learn(x_train, y_train, genetic_optimizer, model_ops.get_model_template(), fit_start=True, batch_size=0.1, auto_normalization=False, gen=50, pop_size=100)

test_model = nn.evaluate(x_test, y_test, model=model, show_report=True)