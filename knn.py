import scipy.spatial.distance as distance
import scipy.stats as stats
import numpy as np

class Knn:
    def __init__(self, k, metric):
        self.k = k
        self.metric = metric
    
    # calculate distance between X_train and X_test
    def neighbors(self, X_train, X_test) -> np.ndarray:
        metric = self.metric
        k = self.k

        dist = distance.cdist(X_test, X_train, metric) 
        neighbors_index = np.argpartition(dist, k)[:,:k]
        return neighbors_index

    def predict(self, X_train, X_test, y) -> np.ndarray:

        # Extract the index of the closest k points
        neighbors_index = self.neighbors(X_test, X_train)
        # unique array and an array in which each data label is converted to an index in the unique array
        labels, y_labels = np.unique(y, return_inverse=True)
        # Find mode with stats.mode (majority vote). _ is the number of counts of mode
        label_index, _ = stats.mode(y_labels[neighbors_index], axis=1, keepdims=True)
        # Extract elements by specifying an axis
        pred = labels.take(label_index).ravel()
        return pred
