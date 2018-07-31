import numpy as np


class PCA:
    def __init__(self, k):
        self.k = k
        self.components = None

    def fit(self, data):
        """
        finds best params for X = Mu + A * Lambda
        :param data: data of shape (number of samples, number of features)
        HINT! use SVD
        """
        _,s,v = np.linalg.svd(data)
        eigenvalues = s**2
        biggest_indices = eigenvalues.argsort()[-self.k:][::-1]
        self.components = v[biggest_indices]
        
    def transform(self, data):
        """
        for given data returns Lambdas
        x_i = mu + A dot lambda_i
        where mu is location_, A is matrix_ and lambdas are projection of x_i
        on linear space from A's rows as basis
        :param data: data of shape (number of samples, number of features)
        """
        # Lemma: x is vector and A dot A.T == I, then x's coordinates in Linear Space(A's rows as basis)
        # is A dot x
        lambdas = np.dot(self.components, data.T).T
        return lambdas

    def inverse_transform(self, transformed_data):
        return np.dot(transformed_data, self.components)

    def return_components(self):
        return self.components
