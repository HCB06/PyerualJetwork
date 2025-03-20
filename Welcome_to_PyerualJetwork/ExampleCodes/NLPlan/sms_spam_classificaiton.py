from pyerualjetwork import neu_cpu, ene_cpu, data_operations_cpu, model_operations_cpu
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('spam_dataset.csv')


X = df['message_content']
y = df['is_spam']


vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(X)

X = X.toarray()

# Veriyi eğitim ve test setlerine ayırma
x_train, x_test, y_train, y_test = data_operations_cpu.split(X, y, 0.4, 42)

# One-hot encoding
y_train, y_test = data_operations_cpu.encode_one_hot(y_train, y_test)

# Veri dengeleme
x_train, y_train = data_operations_cpu.synthetic_augmentation(x_train, y_train)

# Ölçekleme
scaler_params, x_train, x_test = data_operations_cpu.standard_scaler(x_train, x_test)

# neu Modeli
# Configuring optimizer
genetic_optimizer = lambda *args, **kwargs: ene_cpu.evolver(*args, **kwargs)
model = neu_cpu.learn(x_train, y_train, genetic_optimizer, fit_start=True, neurons_history=True, target_acc=1)

# Modeli test etme
test_model = neu_cpu.evaluate(x_test, y_test, W=model[model_operations_cpu.get_weights()], show_metrics=True, activation_potentiation=model[model_operations_cpu.get_act()])
