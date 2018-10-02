import matplotlib.pyplot as plt



year = [56.4, 60, 71, 82]
emnb = [44.91, 58.09, 66, 72]
kmeans = [33, 35.4, 41.3, 32.34]
knn = [43.4, 45.3, 51.3, 65.23]
nb = [53.2, 59, 68.3, 78.2]
plt.plot(year, emnb, color='g',label='EM-NB')
plt.plot(year, kmeans, color='orange',label='KMEANS')
plt.plot(year, knn, color='b',label='KNN')
plt.plot(year, nb, color='r',label='NB')
plt.legend(loc='upper left')
plt.xlabel('Cantidad de documentos')
plt.ylabel('Efectividad  ( % )')
plt.title('Analiis de sentimiento')
plt.show(block=True)