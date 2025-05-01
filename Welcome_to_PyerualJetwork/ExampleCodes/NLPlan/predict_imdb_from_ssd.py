import numpy as np
from pyerualjetwork import model_ops
from pyerualjetwork.cpu import activation_functions
import pickle

review = ["very bad, too bad discusting!, like a shit, it is poor casting and poor film with a really bad and worst scenario"]

"""
exmpl negative review: I had high hopes for this film, but unfortunately, it was a huge letdown. The plot was all over the place, lacking any real direction or depth.
The characters felt one-dimensional and unrelatable, making it hard to care about what happened to them.
The pacing was painfully slow, with unnecessary scenes that dragged on for far too long.
Even the performances felt uninspired, as if the actors were just going through the motions.
Visually, it was dull, and the soundtrack did nothing to enhance the experience.
Overall, it felt like a wasted opportunity with little to offer.



exmpl positive review: This movie was an absolute delight from start to finish! The storytelling was captivating, with well-developed characters that felt truly authentic.
The cinematography was stunning, beautifully capturing every emotion and detail.
The performances were top-notch, especially from the lead actor, who brought so much depth to their role.
The soundtrack perfectly complemented the mood of each scene, elevating the entire experience.
It's one of those films that leaves you thinking long after it's over, with a perfect balance of drama, humor, and heartfelt moments.
Definitely a must-watch!


"""

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

inp_vectorized = vectorizer.transform(review)

inp = inp_vectorized.toarray()

# Model ile tahmin yapma
predict = model_ops.predict_from_storage(inp, model_name='IMDB', model_path='')

# Tahmini yorumlama
if np.argmax(predict) == 1:
    predict_label = 'pozitif'
elif np.argmax(predict) == 0:
    predict_label = 'negatif'

print('%' + str(int(max(activation_functions.Softmax(predict) * 100))) + ' ' + predict_label)