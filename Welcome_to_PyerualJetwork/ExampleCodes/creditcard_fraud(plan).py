import pandas as pd
from pyerualjetwork import neu_cpu, ene_cpu, data_operations_cpu, model_operations_cpu

data = pd.read_csv('creditcard.csv') # dataset link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

x = data.drop('Class', axis=1).values
y = data['Class'].values

x = data_operations_cpu.normalization(x)
x_train, x_test, y_train, y_test = data_operations_cpu.split(x, y, 0.4, 42)

y_train, y_test = data_operations_cpu.encode_one_hot(y_train, y_test)

scaler_params, x_train, x_test = data_operations_cpu.standard_scaler(x_train, x_test)

x_train, y_train = data_operations_cpu.synthetic_augmentation(x_train, y_train)

# Configuring optimizer
genetic_optimizer = lambda *args, **kwargs: ene_cpu.evolver(*args, **kwargs)
model = neu_cpu.learn(x_train, y_train, genetic_optimizer, fit_start=True, batch_size=0.1, auto_normalization=False) # learner function = TFL(Test Feedback Learning). If test parameters not given then uses Train Feedback. More information: https://github.com/HCB06/pyerualjetwork/blob/main/Welcome_to_neu/neu.pdf

activation_potentiation = model[model_operations_cpu.get_act()]
W = model[model_operations_cpu.get_weights()]

test_model = neu_cpu.evaluate(x_test, y_test, show_metrics=True, W=W, activations=activation_potentiation)