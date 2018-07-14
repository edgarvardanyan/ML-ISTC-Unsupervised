import numpy as np
from collections import Counter

class K_NN(object):
    def __init__(self, k):
        """
        :param k: number of nearest neighbours
        """
        self.k = k

    def fit(self, data):
        """
        :param data: 3D array, where data[i, j] is i-th classes j-th point (vector: D dimenstions)
        """
        self.data = np.array(data)
        pass

    def predict(self, data):
        """
        :param data: 2D array of floats N points each D dimensions
        :return: array of integers
        """
        data = np.array(data)
        distances = np.empty((data.shape[0], self.data.shape[0] *  self.data.shape[1])).astype(object)
        for k in range(data.shape[0]):
            for i in range(self.data.shape[0]):
                for j in range(self.data.shape[1]):
                    distances[k][i*self.data.shape[1] + j] = (self.distance(data[k], self.data[i][j]), i)
        distances.sort(axis = 1)
        a = np.empty((data.shape[0], self.k)).astype(object)
        classes = np.empty(data.shape[0]).astype(object)
        for i in range(data.shape[0]):
            for j in range(self.k):
                a[i][j] = distances[i][j][1]
            dict1 = Counter(a[i])
            classes[i] = max(dict1, key = dict1.get)
            
        return classes

    def distance(self, a, b):
        return np.sum((a-b)**2)**0.5