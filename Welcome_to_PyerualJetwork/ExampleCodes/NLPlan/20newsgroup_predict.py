import numpy as np
from pyerualjetwork.cpu import model_ops, activation_functions
import pickle

with open('tfidf_20news.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

model = model_ops.load_model(model_name='20newsgroup', model_path='')


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
        if np.argmax(predict) == 0:
            predict_label = 'Alternatif Ateizm'
        elif np.argmax(predict) == 1:
            predict_label = 'Bilgisayar Grafikleri'
        elif np.argmax(predict) == 2:
            predict_label = 'Windows İşletim Sistemi'
        elif np.argmax(predict) == 3:
            predict_label = 'IBM PC Donanımı'
        elif np.argmax(predict) == 4:
            predict_label = 'Mac Donanımı'
        elif np.argmax(predict) == 5:
            predict_label = 'X Penceresi Sistemi'
        elif np.argmax(predict) == 6:
            predict_label = 'Satış İlanları'
        elif np.argmax(predict) == 7:
            predict_label = 'Arabalar'
        elif np.argmax(predict) == 8:
            predict_label = 'Motosikletler'
        elif np.argmax(predict) == 9:
            predict_label = 'Baseball Sporları'
        elif np.argmax(predict) == 10:
            predict_label = 'Hokey Sporları'
        elif np.argmax(predict) == 11:
            predict_label = 'Kriptografi'
        elif np.argmax(predict) == 12:
            predict_label = 'Elektronik Bilimi'
        elif np.argmax(predict) == 13:
            predict_label = 'Tıp Bilimi'
        elif np.argmax(predict) == 14:
            predict_label = 'Uzay Bilimi'
        elif np.argmax(predict) == 15:
            predict_label = 'Hristiyanlık'
        elif np.argmax(predict) == 16:
            predict_label = 'Silah Politikaları'
        elif np.argmax(predict) == 17:
            predict_label = 'Orta Doğu Siyaseti'
        elif np.argmax(predict) == 18:
            predict_label = 'Genel Siyaset Tartışmaları'
        elif np.argmax(predict) == 19:
            predict_label = 'Din ve Diğer Tartışmalar'
            

        print('%' + str(int(max(activation_functions.Softmax(predict) * 100))) + ' ' + predict_label + '\n')

    except:

        pass
