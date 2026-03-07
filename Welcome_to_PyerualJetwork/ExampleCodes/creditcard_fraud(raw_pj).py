import pandas as pd
from pyerualjetwork.cpu import data_ops
from pyerualjetwork import nn, ene, model_ops

data = pd.read_csv('creditcard.csv') # dataset link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

x = data.drop('Class', axis=1).values
y = data['Class'].values

x = data_ops.normalization(x)

x_train, x_test, y_train, y_test = data_ops.split(x, y, 0.4, 42)
y_train, y_test = data_ops.encode_one_hot(y_train, y_test)
x_train, y_train = data_ops.synthetic_augmentation(x_train, y_train)
# Configuring optimizer

gradient_optimizer = lambda *args, **kwargs: nn.grad(*args, method='adam', **kwargs)

model = nn.learn(x_train, y_train, gradient_optimizer=gradient_optimizer, fit_start=False, neurons=[64], activation_functions=['relu'], loss='categorical_corssentropy', backprop_train=True, iter=40)
test_model = nn.evaluate(x_test, y_test, model, show_report=True)

print('Test ACC:', test_model[model_ops.get_acc()])

model_ops.save_model(model, 'credit_card_ptnn')


