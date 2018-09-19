import pandas as pd
import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import roc_auc_score
from tkinter import *

df = pd.read_csv("./final/data/rest_class.1.txt", sep=",",
                 names=["liked", "txt", "date"])
stopsetset = set(stopwords.words('spanish'))
stopsetset.add('uber')
stopsetset.add('Uber_peru')
stopsetset.add('uber_peru')
stopsetset.add('username')
stopsetset.add('hashtag')
stopsetset.add('url')
stopsetset.add('taxi')
stopsetset.add('emoji')

vectorizer = TfidfVectorizer(
    use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopsetset)
y = df.liked
X = vectorizer.fit_transform(df.txt).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=30)
clf = BayesianGaussianMixture(n_components=2,covariance_type='full')
clf.fit(X)
print(clf.predict(X))

root = Tk()
root.wm_title('ANALISIS DE SENTIMIENTO')
top_frame = Frame(root)
top_frame.pack()
bottom_frame = Frame(root)
bottom_frame.pack(side=BOTTOM)
l1 = Label(top_frame, text='Esribe un texto:')
l1.pack(side=LEFT)
w = Text(top_frame, height=3)
w.pack(side=LEFT)

# clf = get_classifier()

def main_op():
    texto = (w.get('1.0', END))
    tw_array = np.array([texto])
    tw_array_vector = vectorizer.transform(tw_array).toarray()
    valor = clf.predict(tw_array_vector)
    print(valor)
    demo2 = ('El exto es :  ' + ("positivo" if valor[0] == 1 else "negativo"))
    l2 = Label(bottom_frame, text=demo2)
    l2.pack()


button = Button(bottom_frame, text='Analizar', command=main_op)
button.pack(side=BOTTOM)

root.mainloop()
