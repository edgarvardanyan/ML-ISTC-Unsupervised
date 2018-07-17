import numpy as np
import random as rand
from numpy.random import choice

class KMeans:
    def __init__(self, k):
        self.k = k

    def fit(self, data):
        """
        :param data: numpy array of shape (k, ..., dims)
        """
        self.dim = data.shape[-1]
        data = np.reshape(data, (-1, self.dim))
        self._initialize_means(data)
        while True:
            classes = np.empty((data.shape[0])).astype(object)
            for i in range(data.shape[0]):
                min_dist = float('inf')
                for j in range(self.k):
                    curr_dist = self.distance(data[i], self.means[j])
                    if curr_dist <min_dist:
                        classes[i] = j
                        min_dist = curr_dist
            class_count = np.zeros((self.k)).astype(float)
            for i in range(data.shape[0]):
                class_count[classes[i]] += 1
            new_means = np.zeros((self.k, self.dim)).astype(float)
            for i in range(data.shape[0]):
                new_means[classes[i]] += data[i]/class_count[classes[i]]
            if (np.sum((new_means-self.means)**2)/self.dim)**0.5<((data.max()-data.min())/1000):
                self.classes = classes
                break
            else:
                self.means = new_means

    def _initialize_means(self, data):
        centers = rand.sample(range(0, data.shape[0]), self.k)
        self.means = np.zeros((self.k, data.shape[1]))
        for i in range(self.k):
            self.means[i] = data[centers[i]]

    def predict(self, data):
        """
        :param data: numpy array of shape (k, ..., dims)
        :return: labels of each datapoint and it's mean
                 0 <= labels[i] <= k - 1
        """
        data = np.reshape(data, (-1, self.dim))
        classes = np.empty((data.shape[0])).astype(int)
        for i in range(data.shape[0]):
            min_dist = float('inf')
            for j in range(self.k):
                curr_dist = self.distance(data[i], self.means[j])
                if curr_dist<min_dist:
                    classes[i] = j
                    min_dist = curr_dist
        means = self.means[classes]
        return classes, means
    
    def distance(self, vect1, vect2):
        return np.sum((vect1-vect2)**2)**0.5

class KMeansPlusPlus(KMeans):
    def _initialize_means(self, data):
        centers = np.zeros((self.k,)).astype(int)
        centers[0] = int(choice(range(0, data.shape[0])))
        for i in range(1, self.k):
            weights = np.zeros((data.shape[0],))
            for k in range(data.shape[0]):
                min_dist = float('inf')
                for j in range(i):
                    curr_dist = self.distance(data[centers[j]], data[k])
                    if curr_dist<min_dist:
                        min_dist = curr_dist
                weights[k] = min_dist**2
            weights = weights/np.sum(weights)
            centers[i] = choice(range(0, data.shape[0]), p = weights)
        self.means = data[centers]
