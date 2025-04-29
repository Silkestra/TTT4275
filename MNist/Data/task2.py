from scipy.io import loadmat
import numpy as np 
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt 
import time 
from sklearn.cluster import KMeans
from sklearn import metrics
import seaborn as sn
from collections import Counter



data = loadmat('TTT4275/MNist/Data/data_all.mat')
training_vector = data['trainv']
training_labels = data['trainlab'].ravel()
test_vector = data['testv']
test_labels = data['testlab'].ravel()
num_train = int(data['num_train'].item())
num_test  = int(data['num_test'].item())
row_size  = int(data['row_size'].item())
col_size  = int(data['col_size'].item())
vec_size  = int(data['vec_size'].item())
num_classes = 10 



def NN_classifier(chunk_size):
    start_time = time.time()
    total_prediction = np.empty(num_test)

    for i in range(0, num_test, chunk_size):
        chunk_end = min(i + chunk_size,num_test)
        test_vector_chunk = test_vector[i:chunk_end]

        distance_matrix = cdist(test_vector_chunk, training_vector, metric='euclidean')
        prediction_vector = np.argmin(distance_matrix, axis=1)
        predicted_numbers = training_labels[prediction_vector]
        total_prediction[i:chunk_end] = predicted_numbers

    correct_predictions = (total_prediction == test_labels)

    error_rate = 1 - np.sum(correct_predictions)/num_test
    print("Runtime NN_classifier:", time.time() - start_time)
    return total_prediction, error_rate


def clustering(M):
    cluster_start_time = time.time()

    class_centres = np.zeros((num_classes, M, training_vector.shape[1]), dtype=training_vector.dtype)

    for cl in range(num_classes):
        class_i = training_vector[training_labels == cl]
        kmeans = KMeans(n_clusters=M,random_state=42)
        id_xi = kmeans.fit_predict(class_i)
        class_centres[cl] = kmeans.cluster_centers_

    templates = class_centres.reshape(-1, training_vector.shape[1])     #Er her Cl x M x 784 
    template_labels = np.repeat(np.arange(num_classes), M)

    cluster_end_time = time.time()
    print("Clustering duration:", cluster_end_time - cluster_start_time)
    return templates.reshape(-1, 784), template_labels 


def cluster_NN_classifier(M, chunk_size):  # M er templates per klasse
    start_time = time.time()
    total_prediction = np.empty(num_test)
    
    templates, template_labels = clustering(M)
    for i in range(0, num_test, chunk_size):
        chunk_end = min(i + chunk_size,num_test)
        test_vector_chunk = test_vector[i:chunk_end]

        distance_matrix = cdist(test_vector_chunk, templates, metric='euclidean')
        prediction_vector = np.argmin(distance_matrix, axis=1)
        predicted_numbers = template_labels[prediction_vector]
        total_prediction[i:chunk_end] = predicted_numbers

    correct_predictions = (total_prediction == test_labels)
    error_rate = 1 - np.sum(correct_predictions)/num_test
    print("Runtime cluster_NN_classifier:", time.time() - start_time)
    return total_prediction, error_rate, correct_predictions


def cluster_KNN_classifier(M,chunk_size, K):
    start_time = time.time()
    templates, template_labels = clustering(M)

    total_prediction = np.empty(num_test)
    for i in range(0, num_test, chunk_size):
        chunk_end = min(i + chunk_size,num_test)
        test_vector_chunk = test_vector[i:chunk_end]

        distance_matrix = cdist(test_vector_chunk, templates, metric='euclidean')
        k_nearest = np.argsort(distance_matrix)[:,:K]
        k_nearest_labels = template_labels[k_nearest]
        predictions = np.array([Counter(row).most_common(1)[0][0] for row in k_nearest_labels])

        total_prediction[i:chunk_end] = predictions

    correct_predictions = (total_prediction == test_labels)
    error_rate = 1 - np.sum(correct_predictions)/num_test
    print("Runtime cluster_KNN_classifier:", time.time() - start_time)
    return total_prediction, error_rate, correct_predictions 


def plotConfusionMatrix(cm, classes, title=""):
    
    fig = plt.figure(figsize=(6,4))
    sn.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes,
                yticklabels=classes)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    if title != "":
        plt.title(title)
    return fig

## Saving function
def saveFigsAsPDF(figures, names: list[str], task: str, dirr="Plots/"):
    saveNames = [task + "_" + name + ".pdf" for name in names]

    for fig, saveName in zip(figures, saveNames):
        fig.savefig(dirr + saveName)


def task1(save=False):
    total_prediction, error_rate, correct_predictions = NN_classifier(1000)
    cm = metrics.confusion_matrix(test_labels, total_prediction)
    classes = [str(i) for i in range(10)]
    confusion_matrix = plotConfusionMatrix(cm, classes, 'Error rate: {(error_rate * 100):.2} %')
    plt.show()


    if save:
        figs = [confusion_matrix,]    
        # Name for figures when saving 
        plotNames = ["C"]
        saveFigsAsPDF(figs, plotNames, "task1")
    return


#For error rate ved 1-201 clusters 

k_values = []
error_rates = []

""" # Loop over k from 1 to 200
for k in range(1, 101, 10):
    predicted, error_rate = cluster_KNN_classifier(k, 1000, 7)  # Keep 1000 and 7 fixed
    k_values.append(k)
    error_rates.append(error_rate)

# Plotting
plt.figure(figsize=(10,6))
plt.plot(k_values, error_rates, marker='o')
plt.title('KNN Error Rate vs. k')
plt.xlabel('k (Number of neighbors)')
plt.ylabel('Error Rate')
plt.grid(True)
plt.show() """



""" prediction_failed_idx = np.where(total_prediction != test_labels)[0]
    prediction_correct_idx = np.where(total_prediction == test_labels)[0]

    #Alternativ for å finne prediction miss
    prediction_miss = []
    for i in range(num_test):          
        print(i)
        if total_prediction[i] != test_labels[i]:
            prediction_miss.append(i)  

    test_failed_predictions = test_vector[prediction_correct_idx
    test_correct_predictions = test_vector[prediction_correct_idx[0]].reshape((28, 28))
    error_rate = np.sum(correct_predictions)/num_test
 """