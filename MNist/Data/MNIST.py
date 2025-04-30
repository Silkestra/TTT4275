import numpy as np 
from scipy.io import loadmat
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt 
import time 
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn import metrics
from sklearn.datasets import make_blobs
import seaborn as sn
from collections import Counter



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
    return total_prediction, error_rate, correct_predictions

def KNN_classifier(chunk_size, K):
    start_time = time.time()
    total_prediction = np.empty(num_test)
    for i in range(0, num_test, chunk_size):
        chunk_end = min(i + chunk_size,num_test)
        test_vector_chunk = test_vector[i:chunk_end]

        distance_matrix = cdist(test_vector_chunk, training_vector, metric='euclidean')
        k_nearest = np.argsort(distance_matrix)[:,:K]
        k_nearest_labels = training_labels[k_nearest]
        predictions = np.array([Counter(row).most_common(1)[0][0] for row in k_nearest_labels])

        total_prediction[i:chunk_end] = predictions

    correct_predictions = (total_prediction == test_labels)
    error_rate = 1 - np.sum(correct_predictions)/num_test
    print("Runtime KNN_classifier:", time.time() - start_time)
    return total_prediction, error_rate, correct_predictions

def clustering(M):
    cluster_start_time = time.time()

    class_centres = np.zeros((num_classes, M, training_vector.shape[1]), dtype=training_vector.dtype)

    for cl in range(num_classes):
        class_i = training_vector[training_labels == cl]
        kmeans = KMeans(n_clusters=M,random_state=42)
        id_xi = kmeans.fit_predict(class_i)
        class_centres[cl] = kmeans.cluster_centers_

    templates = class_centres.reshape(-1, training_vector.shape[1])  
    template_labels = np.repeat(np.arange(num_classes), M)

    cluster_end_time = time.time()
    print("Clustering duration:", cluster_end_time - cluster_start_time)
    return templates.reshape(-1, 784), template_labels 


def cluster_NN_classifier(M, chunk_size): 
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


### Plot and save functions ###

def plotConfusionMatrix(cm, classes, title=""):
    
    fig = plt.figure(figsize=(6,4))
    sn.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes,
                yticklabels=classes)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    if title != "":
        plt.title(title)
    return fig

def saveFigsAsPDF(figures, names: list[str], task: str, dirr="MNist/Data/Plots/"):
    saveNames = [task + "_" + name + ".pdf" for name in names]

    for fig, saveName in zip(figures, saveNames):
        fig.savefig(dirr + saveName)



data = loadmat('MNist/Data/data_all.mat')
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


### Task functions ###    

def task1(save=False):
    total_prediction, error_rate, correct_predictions = NN_classifier(1000)
    
    cm = metrics.confusion_matrix(test_labels, total_prediction)
    classes = [str(i) for i in range(10)]
    confusion_matrix = plotConfusionMatrix(cm, classes, f"Confusion matrix NN without clustering\nChunksize: 1000, Test size: 10000\nError rate = {error_rate * 100:.1f}%")
    # plt.show()

    misclassified_idx = np.where(total_prediction != test_labels)[0][:3]
    correctly_classified_idx = np.where(total_prediction == test_labels)[0][:3]
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    fig.suptitle('First 3 Misclassified and Correctly Classified Images')

    for i, idx in enumerate(misclassified_idx):
        ax = axes[0, i]
        ax.imshow(test_vector[idx].reshape(28, 28), cmap='gray')
        ax.axis('off')
        ax.set_title(f'True: {test_labels[idx]}, Pred: {int(total_prediction[idx])}')

    for i, idx in enumerate(correctly_classified_idx):
        ax = axes[1, i]
        ax.imshow(test_vector[idx].reshape(28, 28), cmap='gray')
        ax.axis('off')
        ax.set_title(f'True: {test_labels[idx]}')
    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    # plt.show()


    if save:
        figs = [confusion_matrix, fig]    
        plotNames = ["ConfusionMatrix", "MisclassifiedExamples"]
        saveFigsAsPDF(figs, plotNames, "task1")
    return

def task2b(save=False):
    total_prediction, error_rate, correct_predictions = cluster_NN_classifier(64,1000)
    cm = metrics.confusion_matrix(test_labels, total_prediction)
    classes = [str(i) for i in range(10)]
    confusion_matrix = plotConfusionMatrix(cm, classes, f"Confusion matrix NN with clustering\nChunksize: 1000, Test size: 10000\nError rate = {error_rate * 100:.1f}%")
    # plt.show()

    if save:
        figs = [confusion_matrix,]    
        # Name for figures when saving 
        plotNames = ["C"]
        saveFigsAsPDF(figs, plotNames, "task2b")
    return

def task2c(save=False):
    total_prediction, error_rate, correct_predictions = cluster_KNN_classifier(64,1000,7)
    cm = metrics.confusion_matrix(test_labels, total_prediction)
    classes = [str(i) for i in range(10)]
    confusion_matrix = plotConfusionMatrix(cm, classes, f"Confusion matrix 7-NN with clustering\nChunksize: 1000, Test size: 10000\nError rate = {error_rate * 100:.1f}%")
    # plt.show()

    if save:
        figs = [confusion_matrix,]    
        # Name for figures when saving 
        plotNames = ["C"]
        saveFigsAsPDF(figs, plotNames, "task2c")
    return

def plot_KNN_without_clustering(save=False):
    total_prediction, error_rate, correct_predictions = KNN_classifier(1000,7)
    cm = metrics.confusion_matrix(test_labels, total_prediction)
    classes = [str(i) for i in range(10)]
    confusion_matrix = plotConfusionMatrix(cm, classes, f"Confusion matrix 7-NN without clustering\nChunksize: 1000, Test size: 10000\nError rate = {error_rate * 100:.1f}%")
    # plt.show()

    if save:
        figs = [confusion_matrix,]    
        # Name for figures when saving 
        plotNames = [""]
        saveFigsAsPDF(figs, plotNames, "extra: KNN without clustering")
    return

def cluster_centroids_example(save = False):
    X, y = make_blobs(n_samples=500, centers=4, cluster_std=0.6, random_state=40)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    init_bad = np.array([[10, 10], [10.5, 10.5], [11, 11], [11.5, 11.5]])
    kmeans_bad = MiniBatchKMeans(n_clusters=4, init=init_bad, n_init=1, batch_size=100, random_state=42)
    kmeans_bad.partial_fit(X)
    ax1.scatter(X[:, 0], X[:, 1], alpha=0.3)
    ax1.scatter(kmeans_bad.cluster_centers_[:, 0], kmeans_bad.cluster_centers_[:, 1], marker='x', s=100, c='black', label='Init')
    for i in range(5):
        kmeans_bad.partial_fit(X)
        ax1.scatter(kmeans_bad.cluster_centers_[:, 0], kmeans_bad.cluster_centers_[:, 1], marker='x', s=100, label=f'Iter {i+1}')
    ax1.set_title("Convergence from Unfavorable Initial Centroids")
    ax1.legend()

    kmeans_good = MiniBatchKMeans(n_clusters=4, init='k-means++', n_init=1, batch_size=100, random_state=42)
    kmeans_good.partial_fit(X)
    ax2.scatter(X[:, 0], X[:, 1], alpha=0.3)
    ax2.scatter(kmeans_good.cluster_centers_[:, 0], kmeans_good.cluster_centers_[:, 1], marker='x', s=100, c='black', label='Init')
    for i in range(5):
        kmeans_good.partial_fit(X)
        ax2.scatter(kmeans_good.cluster_centers_[:, 0], kmeans_good.cluster_centers_[:, 1], marker='x', s=100, label=f'Iter {i+1}')
    ax2.set_title("Convergence from Favorable Initial Centroids")
    ax2.legend()

    fig.suptitle("Effect of Initial Centroids on K-means Convergence", fontsize=14)
    plt.tight_layout()
    # plt.show()

    if save:
        figs = [fig]     
        plotNames = [""]
        saveFigsAsPDF(figs, plotNames, "cluster_centroids_example2")
    return


def clustering_silhouette(save=False):
    M_values = range(2, 100, 2)
    silhouette_scores = []

    for M in M_values:
        print(f"Clustering with M = {M}")
        templates, template_labels = clustering(M)
        try:
            score = silhouette_score(templates, template_labels)
            silhouette_scores.append(score)
        except ValueError as e:
            print(f"Could not compute silhouette for M = {M}: {e}")
            silhouette_scores.append(float('nan'))

    fig = plt.figure(figsize=(10, 6)) 
    plt.plot(M_values, silhouette_scores, marker='o')
    plt.title("Silhouette Score as a Function of Clusters per Class")
    plt.xlabel("M (Clusters per Class)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    # plt.show()

    if save:
        figs = [fig]     
        plotNames = [""]
        saveFigsAsPDF(figs, plotNames, "cluster_silhouette")
    return

def clustering_count_error_rate(save=False):
    k_values = []
    error_rates = []

    for k in range(1, 100, 2):
        predicted, error_rate, pr = cluster_KNN_classifier(k, 1000, 7) 
        k_values.append(k)
        error_rates.append(error_rate)

    fig = plt.figure(figsize=(10, 6)) 
    plt.plot(k_values, error_rates, marker='o')
    plt.title('Effect of Cluster Count per Class on KNN Classification Error')
    plt.xlabel('k (Number of clusters per class)')
    plt.ylabel('Error Rate')
    plt.grid(True)
    # plt.show()

    if save:
        figs = [fig]     
        plotNames = [""]
        saveFigsAsPDF(figs, plotNames, "cluster_count")
    return




### Task selection ###  

task1(save=True)
task2b(save=True)
task2c(save=True)
# Additional result
plot_KNN_without_clustering(save=True)


## Figures used in discussion
cluster_centroids_example(save=True)
clustering_count_error_rate(save=True)
clustering_silhouette(save=True)

plt.show()