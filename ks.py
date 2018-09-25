# Importing libraries
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.metrics import accuracy_score
 
df = pd.read_csv("./final/data/rest_class.txt", sep=",",
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
X = vectorizer.fit_transform(df.txt)
# Creating true labels for 30 training sentences
y = df.liked
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# Clustering the document with KNN classifier
modelknn = KNeighborsClassifier(n_neighbors=2)
modelknn.fit(X_train,y_train)
print(modelknn.score(X_test,y_test))

# Clustering the training 30 sentences with K-means technique
modelkmeans = KMeans(n_clusters=2, init='k-means++', max_iter=200, n_init=1)
modelkmeans.fit(X_train)


# Predicting it on test data : Testing Phase
test_sentences = ["rosario  tomarse patente igj 654 3000 tiro trabo puertas",\
"llegando voy caerr", "pequea obsesion cancion historia ricardo arjona"]

Test = vectorizer.transform(test_sentences)
 
true_test_labels = ['positive','negative']
predicted_labels_knn = modelknn.predict(Test)
predicted_labels_kmeans = modelkmeans.predict(X_test)
print(accuracy_score(y_test, predicted_labels_kmeans))

print("########--------KMEANS -------########")
print(predicted_labels_kmeans)
print("########--------KNN -------########")
print(predicted_labels_knn)