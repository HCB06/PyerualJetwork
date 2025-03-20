import numpy as np
from pyerualjetwork import model_operations_cpu, activation_functions_cpu
import pickle

with open('tfidf_imdb.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

model = model_operations_cpu.load_model(model_name='IMDB', model_path='')

while True:

    text = input()

    if text == '':

        text = None

    try:

        text_news = [text]

        inp_vectorized = vectorizer.transform(text_news)

        inp = inp_vectorized.toarray()

        # Model ile tahmin yapma
        predict = model_operations_cpu.predict_from_memory(Input=inp, W=model[model_operations_cpu.get_weights()], activations=model[model_operations_cpu.get_act()], scaler_params=model[model_operations_cpu.get_scaler()])


        # Tahmini yorumlama
        if np.argmax(predict) == 1:
            predict_label = 'pozitif'
        elif np.argmax(predict) == 0:
            predict_label = 'negatif'

        print('%' + str(int(max(activation_functions_cpu.Softmax(predict) * 100))) + ' ' + predict_label + '\n')

    except:

        pass
