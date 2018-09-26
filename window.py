import pandas as pd
import nltk
import numpy as np
from nltk.corpus import stopwords
import time 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from tkinter import *

df_original = pd.read_csv("./data/data.csv", sep=",",
                 names=["polaridad", "texto"])
df_clean = pd.read_csv("./data/clean.txt", sep=",",
                 names=["polaridad", "texto"])

df = pd.read_csv("./data/clean.txt", sep=",",
                 names=["polaridad", "texto"])
stopsetset = set(stopwords.words('spanish'))
stopsetset.add('uber')
stopsetset.add('Uber_peru')
stopsetset.add('uber_peru')
stopsetset.add('username')
stopsetset.add('hashtag')
stopsetset.add('url')
stopsetset.add('taxi')
stopsetset.add('emoji')

root = Tk()
root.geometry('1200x600')
root.title("Registration Form")
texto = StringVar()

def procesando(p):
    texto = "Procesando...."  if(p) else ""
    res_p.config(text=texto) 

def extraer():
    # entry_one.insert(INSERT, texto.get())
    entry_one.insert(INSERT, str(df_original.texto[0:100]))
    print("Extraer")

def limpiar():
    entry_two.insert(INSERT, str(df_clean.texto[0:100]))
    print("Limpiando")

def procesar():
    procesando(True)
    nb_accurrency = nb()
    knn_accurrency = knn()
    kmeans_accurrency = kmeans()
    emnb_accurrency = emnb()
    calcular_c(nb_accurrency , knn_accurrency, kmeans_accurrency,emnb_accurrency)
    print("Procesado")
    procesando(False)

def calcular():
    res_one.config(text='Navies Bayes - EM : Positivo')
    res_two.config(text='Navies Bayes : Negativo')
    res_tree.config(text='KMeans : Positivo')
    res_four.config(text='KNN : Negativo')

def calcular_c(nb, knn, kmeans, emnb):
    res_one_c.config(text='Navies Bayes - EM : Positivo: ' + emnb)
    res_two_c.config(text='Efectvidad de Navies Bayes : ' + nb)
    res_tree_c.config(text='Efectvidad de KMEANS : '+ kmeans)
    res_four_c.config(text='Efectvidad de KNN : '+ knn)

def emnb():
    vectorizer = TfidfVectorizer(
    use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopsetset)
    y = df.polaridad
    X = vectorizer.fit_transform(df.texto.values.astype('U')).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=30)
    clf = BayesianGaussianMixture(n_components=2,covariance_type='full')
    clf.fit(X_train)
    predicted = clf.predict(X_test)
    score = str(accuracy_score(y_test, predicted))
    print("Porcentaje EMNB : " + score)
    return score

def nb():
    vectorizer = TfidfVectorizer(
    use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopsetset)
    y = df.polaridad
    X = vectorizer.fit_transform(df.texto.values.astype('U'))
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=42)
    clf = naive_bayes.MultinomialNB()
    clf.fit(X_train, y_train)
    score = str(roc_auc_score(y_test, clf.predict_log_proba(X_test)[:, 1]))
    print("Porcentaje naives bayes : " + score)
    return score

def knn():
    vectorizer = TfidfVectorizer(
    use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopsetset)
    X = vectorizer.fit_transform(df.texto.values.astype('U'))
    # Creating true labels for 30 training sentences
    y = df.polaridad
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    modelknn = KNeighborsClassifier(n_neighbors=2)
    modelknn.fit(X_train,y_train)
    score = str(modelknn.score(X_test,y_test))
    print("Porcentaje KNN : " + score )
    return score

def kmeans():
    vectorizer = TfidfVectorizer(
    use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopsetset)
    X = vectorizer.fit_transform(df.texto.values.astype('U'))
    # Creating true labels for 30 training sentences
    y = df.polaridad
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    modelkmeans = KMeans(n_clusters=2, init='k-means++', max_iter=200, n_init=1)
    modelkmeans.fit(X_train)
    predicted_labels_kmeans = modelkmeans.predict(X_test)
    score = str(accuracy_score(y_test, predicted_labels_kmeans))
    print("Porcentaje KMEANS : " + score)
    return score


title = Label(root, text="Análisis de sentimiento",width=20,font=("bold", 20))
title.place(x=0,y=0)

Button(root, text='Extraer',width=15,bg='red',fg='white',font=("bold", 10), command=extraer).place(x=10,y=50)
Button(root, text='Limpiar',width=15,bg='blue',fg='white',font=("bold", 10), command=limpiar).place(x=150,y=50)
Button(root, text='Procesar',width=15,bg='green',fg='white',font=("bold", 10), command=procesar).place(x=290,y=50)

entry_one = Text(root, width=80, height=15)
entry_one.place(x=10,y=100)

entry_two = Text(root, width=80, height=15)
entry_two.place(x=10,y=350)


title = Label(root, text="Ingresa texto",width=10,font=("bold", 10))
title.place(x=700,y=100)
Button(root, text='Calcular',width=10,bg='green',fg='white',font=("bold", 10), command=calcular).place(x=800,y=95)

res_one = Label(root, text="Navies Bayes - EM :",width=30,font=("bold", 10))
res_one.place(x=700,y=170)
res_two = Label(root, text="Navies Bayes :",width=30,font=("bold", 10))
res_two.place(x=700,y=200)
res_tree = Label(root, text="KMeans :",width=30,font=("bold", 10))
res_tree.place(x=700,y=230)
res_four = Label(root, text="KNN :",width=30,font=("bold", 10))
res_four.place(x=700,y=260)


res_one_c = Label(root, text="Efectvidad de Navies Bayes - EM :",width=40,font=("bold", 10))
res_one_c.place(x=700,y=350)
res_two_c = Label(root, text="Efectvidad de Navies Bayes :",width=40,font=("bold", 10))
res_two_c.place(x=700,y=380)
res_tree_c = Label(root, text="Efectvidad de KMeans :",width=40,font=("bold", 10))
res_tree_c.place(x=700,y=410)
res_four_c = Label(root, text="Efectvidad de KNN :",width=40,font=("bold", 10))
res_four_c.place(x=700,y=440)

res_p = Label(root, text="",width=40,font=("bold", 10))
res_p.place(x=700,y=470)


entry_texto = Entry(root, width=57, textvariable=texto)
entry_texto.place(x=700,y=130)

root.mainloop()