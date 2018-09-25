import pandas as pd
import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=30)
clf = BayesianGaussianMixture(n_components=2,covariance_type='full')
clf.fit(X_train)
emnb = clf.predict(X_test)
score = str(accuracy_score(y_test, emnb))
print("Porcentaje EM-NB : " + score)
