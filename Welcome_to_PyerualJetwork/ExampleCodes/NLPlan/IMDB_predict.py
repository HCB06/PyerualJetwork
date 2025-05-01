import numpy as np
from pyerualjetwork import model_ops
from pyerualjetwork.cpu import activation_functions
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
        predict = model_ops.predict_from_memory(inp, model)

        # Tahmini yorumlama
        if np.argmax(predict) == 1:
            predict_label = 'pozitif'
        elif np.argmax(predict) == 0:
            predict_label = 'negatif'

        print('%' + str(int(max(activation_functions.Softmax(predict) * 100))) + ' ' + predict_label + '\n')

    except:

        pass
