import random
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from Point import Point
from Cluster import Cluster
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from stop_words import get_stop_words

nltk.download('punkt')

TEST = "./dataSet/test.txt"
PROD = "./dataSet/data.txt"
FILE = "./dataSet/file.txt"
file = open(TEST, "w")
stop_words = get_stop_words("es")
stop_words.remove("no")
stop_words.remove("sí")
NUM_CLUSTERS = 2
ITERATIONS = 1000
COLORS = ['red', 'blue']
tweets = [
    "La aplicación de Taxibeat es una basura",
    "Tienen puros rateros como conductores, una mierda el servicio de Uber",
    "Que estafa los precios de las carreras! 40 soles de La Molina a San Borja",
    "Nuuunca más utilizaré esta aplicación",
    "El taxista nunca llegó y me cobraron la carrera",
    "Cero profesionalismo del conductor que me tocó. Se quedaba dormido",
    "El conductor me cobro extra por 2 cuadras que hijo de puta",
    "Pésima experiencia la que tuve hoy con un conductor de Uber",
    "El peor servicio de taxis que he utilizado",
    "Son unos estafadores! Me cobran adicional por cualquier cosa",
    "Este servicio es un robo mejor es EasyTaxi",
    "Este servicio no es para nada bueno, todo lo contrario",
    "Siempre que pido me cancelan",
    "La peor experiencia de mi vida con este servicio",
    "Una porquería su servicio",
    "Que cagada su aplicación se para colgando",
    "Me da mucha cólera que nunca me toque un buen taxista",
    "Perdí más tiempo esperando al taxi y llegué tarde a mi trabajo",
    "El taxista me cobró doble diciendo que el sistema no aceptaba mi tarjeta... Devuélvanme mi plata!",
    "Voy a denunciarlos por contratar a conductores tan ineptos!! Dioss mio que estrés!",
    "Bueno que puedo decir sobre su servicio.. simplemente un dolor de cabeza",
    "Jamás pensé que habría un servicio peor que TaxiBeat. Hasta que probé Uber. ",
    "No entiendo como calculan sus tarifas, me parece un abuso sus precios.",
    "Con todos estos problemas mejor me tomo mi taxi de la calle",
    "Que horrible olía el carro que me tocó hoy, deberían decirles que se bañen!",
    "Así como están las tarifas me quedaré pobre!",
    "Tanta delincuencia en el país, era obvio que también llegaría a los taxistas...",
    "Después de mi última experiencia me desinstalé la aplicación.",
    "Como ha decaído su servicio, que mal",
    "Sigo sin entender como algunos taxistas tienen buena calificación cuando son un desastre!!",
    "Hoy me dí el susto de mi vida! el taxista cambiaba de ruta constantemente a pesar que le dije que siga el waze",
    "Asi de comunican sus conductores. Ya no son un servicio confiable, voy a eliminar su aplicación llena de mediocres.",
    "mira tú seguridad Uber... He llamado un. Uber y Oh sorpresa..!!! No es el conductor..!!! SEGURIDAD..??",
    "el taxista se aleja del punto de recojo en una señal clara de que no desea hacer el servicio..",
    "Ah y también para reportar a los que te llaman a preguntar a donde vas y después cancelan el viaje ",
    "acá otro chofer que te hace esperar 15 min y luego cancela el servicio.",
    "Cuando tu conductor NO finaliza el viaje y a ti te cobran de más!!",
    "puse más de un destino en la app y se fue de frente a la ultima #wtf ??",
    "Por las puras pongo mi lugar de recojo siempre me hacen caminar hacia el taxi!",
    "por la forma en que te tratan algunos conductores pareciera que no les pagaran.. es el colmo!",
    "estoy INDIGNADA!!!!!!! Su app es totalmente innoperante!!!! Realice un pago pendiente que tenia ya que ustedes me OBLIGARON a introducir una tarjeta de crédito, al querer borrar este medio de pago (ya que yo solo pago con efectivo) y ahora no me deja borrarla!! Exijo ayuda!",
    "El mejor servicio de taxis en Lima, 100% recomendado.",
    "Ayer dejé olvidado mi celular en taxi Uber y me lo devolvieron, Gracias UBER",
    "siempre me salvan cuando estoy tarde para el trabajo, gracias!!",
    "Me encantaron las promociones! saquen más por favoor!",
    "Envié un encargo con uno de sus conductores y llegó al destino sin problemas! Simplemente lo mejor",
    "Me gusta mucho su nuevo logo!",
    "Siempre mando a mi hija en Uber y nunca he tenido ningún problema, todo lo contrario",
    "El conductor que me tocó super educado, incluso me invitó un chocolate! ",
    "Debo reconocer que su servicio no tiene competencia! 10 puntos !!",
    "No puedo estar más satisfecho con ustedes! Simplemente fenomenal!",
    "No saben de todos los apuros que me libran, no sé que haría sin ustedes!",
    "Probablemente el mejor servicio de taxis que existe!",
    "Usando Uber no solo me ahorré más dinero de lo normal sino que llegué más temprano a todos lados!",
    "Si estás tarde para algo siempre puedes confiar en un uber",
    "No podría estar más contento con las facilidades que me da UBER!",
    "Tuve una fea experiencia con EasyTaxi hace un tiempo, pero con Uber me va fenomenal!",
    "Me olvidé mi billetera en un uber el otro día, y que creen.. el conductor me la llevó a mi casa al día siguiente!",
    "Aún cuando estoy ebrio sé que puedo confiar en mis amigos de uber.",
    "Desde que me uní a Uber todo ha sido sensacional",
    "Simplemente genial! La aplicación, el servicio, toodo!",
    "para personas como yo que siempre toman taxi, el sistema perfecto!",
    "No sé si soy yo, pero su sistema cada vez está mejor!",
    "Excelente promoción!! Ahora me sale baritisimo ir a mi casaa",
    "El otro día me fui en pool al trabajo y sinceramente quedé sorprendido! Un gran avance sin duda",
    "No pensé encontrar una aplicación que cumpla con todas mis expectativas! Ustedes si que la rompen",
    "Ayer me equivoqué en el método de pago de mi taxi y el taxista buena gente me espero mientras iba al cajero. Incluso no me cobró adicional! Graciass!!",
]


def get_peso(word):
    words = {
        'basura': -0.7525,
        'mierda': -0.5144,
        'rateros': -0.2645,
        'nunca': -0.7165,
        'cero': -0.5268,
        'hijo de puta': -1,
        'pésima': -1,
        'peor': -1,
        'estafadores': -0.75115,
        'robo': -0.5738,
        'pésimo': -1,
        'pérdida': -0.5743,
        'pesimo': -1,
        'pesima': -1,
        'porquería': -1,
        'cagada': -0.73564,
        'cólera': -0.82121,
        'perdí': -0.5314,
        'perdida': -0.33024,
        'ineptos': -0.42315,
        'dolor': -0.27442,
        'jamás': -0.53115,
        'abuso': -0.91647,
        'horrible': -1,
        'desinstalé': -0.55541,
        'mal': -0.52177,
        'desastre': -0.89314,
        'eliminar': -0.43544,
        'cancelan': -0.45731,
        'cancela': -0.45731,
        'colmo': -0.64516,
        'no': -0.42154,
        'reportar': -0.65314,
        'recomendado': 0.4365,
        'gracias': 0.7156,
        'encantaron': 1,
        'mejor': 0.65647,
        'gusta': 0.54321,
        'super': 0.86558,
        'fenomenal': 1,
        'satisfecho': 0.72015,
        'confiar': 0.54457,
        'contento': 0.66582,
        'súper': 1,
        'recomendar': 0.53475,
        'recomiendo': 0.4618,
        'sensacional': 0.92528,
        'genial': 0.8534,
        'perfecto': 1,
        'avance': 0.43745,
        'rompen': 0.2164,
        'si': 0.4874,
        'buena': 0.43054,
        'bueno': 0.44874,
        'putamadre': 0.3174,
        'ptm': 0.3174,
        'bacán': 0.50314,
        'mejoran': 0.6894,
        'bacan': 0.50314,
        'cheveres': 0.68241,
        'chévere': 0.68241,
        'chevere': 0.68241

    }
    if words.get(word) == None:
        return 0
    else:
        return words.get(word)


def get_tokens(tweet):
    word_tokens = word_tokenize(tweet)
    filtered_tweet = [w for w in word_tokens if not w in stop_words]
    filtered_tweet = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_tweet.append(w)
    return filtered_tweet


def generate_pesos():
    for t in tweets:
        peso = 0
        tokens = get_tokens(t)
        for w in tokens:
            peso += get_peso(w.lower())
        file.write("{}::0.1\n".format(round(peso, 2)))
    file.close()


def dataset_to_list_points(dir_dataset):
    """
    Read a txt file with a set of points and return a list of objects Point
    :param dir_dataset: path file
    """
    generate_pesos()
    points = list()
    with open(dir_dataset, 'rt') as reader:
        for point in reader:
            points.append(
                Point(np.asarray(list(map(float, point.split("::"))))))
    return points


def get_probability_cluster(point, cluster):
    """
    Calculate the probability that the point belongs to the Cluster
    :param point:
    :param cluster:
    :return: probability =
    prob * SUM(e ^ (-1/2 * ((x(i) - mean)^2 / std(i)^2 )) / std(i))
    """
    mean = cluster.mean
    std = cluster.std
    prob = 1.0
    for i in range(point.dimension):
        prob *= (math.exp(-0.5 * (
            math.pow((point.coordinates[i] - mean[i]), 2) /
            math.pow(std[i], 2))) / std[i])

    return cluster.cluster_probability * prob


def get_expecation_cluster(clusters, point):
    """
    Returns the Cluster that has the highest probability of belonging to it
    :param clusters:
    :param point:
    :return: argmax (probability clusters)
    """
    expectation = np.zeros(len(clusters))
    for i, c in enumerate(clusters):
        expectation[i] = get_probability_cluster(point, c)

    return np.argmax(expectation)


def print_clusters_status(it_counter, clusters):
    print('\nITERATION %d' % it_counter)
    for i, c in enumerate(clusters):
        print('\tCluster %d: Probability = %s; Mean = %s; Std = %s;' % (
            i + 1, str(c.cluster_probability), str(c.mean), str(c.std)))


def print_results(clusters):
    print('\n\nFINAL RESULT:')
    for i, c in enumerate(clusters):
        print('\tCluster %d' % (i + 1))
        print('\t\tNumber Points in Cluster: %d' % len(c.points))
        print('\t\tProbability: %s' % str(c.cluster_probability))
        print('\t\tMean: %s' % str(c.mean))
        print('\t\tStandard Desviation: %s' % str(c.std))


def plot_ellipse(center, points, alpha, color):
    """
    Plot the Ellipse that defines the area of Cluster
    :param center:
    :param points: points of cluster
    :param alpha:
    :param color:
    :return: Ellipse
    """

    # Matrix Covariance
    cov = np.cov(points, rowvar=False)

    # eigenvalues and eigenvector of matrix covariance
    eigenvalues, eigenvector = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvector = eigenvector[:, order]

    # Calculate Angle of ellipse
    angle = np.degrees(np.arctan2(*eigenvector[:, 0][::-1]))

    # Calculate with, height
    width, height = 4 * np.sqrt(eigenvalues[order])

    # Ellipse Object
    ellipse = Ellipse(xy=center, width=width, height=height, angle=angle,
                      alpha=alpha, color=color)

    ax = plt.gca()
    ax.add_artist(ellipse)

    return ellipse


def plot_results(clusters):
    plt.plot()
    for i, c in enumerate(clusters):
        # plot points
        x, y = zip(*[p.coordinates for p in c.points])
        plt.plot(x, y, linestyle='None', color=COLORS[i], marker='.')
        # plot centroids
        plt.plot(c.mean[0], c.mean[1], 'o', color=COLORS[i],
                 markeredgecolor='k', markersize=10)
        # plot area
        plot_ellipse(c.mean, [p.coordinates for p in c.points], 0.2, COLORS[i])

    plt.show()


def expectation_maximization(dataset, num_clusters, iterations):
    # Read data set
    points = dataset_to_list_points(dataset)
    # Select N points random to initiacize the N Clusters
    initial = random.sample(points, num_clusters)

    # Create N initial Clusters
    clusters = [Cluster([p], len(initial)) for p in initial]

    # Inicialize list of lists to save the new points of cluster
    new_points_cluster = [[] for i in range(num_clusters)]

    converge = False
    it_counter = 0
    while (not converge) and (it_counter < iterations):
        # Expectation Step
        for p in points:
            i_cluster = get_expecation_cluster(clusters, p)
            new_points_cluster[i_cluster].append(p)

        # Maximization Step
        for i, c in enumerate(clusters):
            c.update_cluster(new_points_cluster[i], len(points))

        # Check that converge all Clusters
        converge = [c.converge for c in clusters].count(False) == 0

        # Increment counter and delete lists of clusters points
        it_counter += 1
        new_points_cluster = [[] for i in range(num_clusters)]

        # Print clusters status
        print_clusters_status(it_counter, clusters)

    # Print final result
    print_results(clusters)

    # Plot Final results
    plot_results(clusters)


if __name__ == '__main__':
    expectation_maximization(FILE, NUM_CLUSTERS, ITERATIONS)
