import numpy as np
from pyerualjetwork.cpu import model_ops, activation_functions
import pickle

with open('tfidf_imdb.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

model = model_ops.load_model(model_name='IMDB', model_path='')

while True:

    text = input()

    if text == '':

        text = None

    try:

        text_news = [text]

        inp_vectorized = vectorizer.transform(text_news)

        inp = inp_vectorized.toarray()

        # Model ile tahmin yapma
        predict = model_ops.predict_from_memory(Input=inp, W=model[model_ops.get_weights()], activations=model[model_ops.get_act()], scaler_params=model[model_ops.get_scaler()])


        # Tahmini yorumlama
        if np.argmax(predict) == 1:
            predict_label = 'pozitif'
        elif np.argmax(predict) == 0:
            predict_label = 'negatif'

        print('%' + str(int(max(activation_functions.Softmax(predict) * 100))) + ' ' + predict_label + '\n')

    except:

        pass
