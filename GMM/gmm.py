import numpy as np
from k_means import KMeans
from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, k):
        self.k = k
        self.means = []
        self.covariances = []
        self.pis = []
        self.gammas = []

    def fit(self, data):
        """
        :params data: np.array of shape (..., dim)
                                  where dim is number of dimensions of point
        """
        self._initialize_params(data)
        old_log = self.log_likelihood(data)
        i = 1
        while True:
            #print (i)
            #print ("Iteration", i, '\n', self.covariances[0], self.means[0], self.pis[0])
            self._E_step(data)
            self._M_step(data)
            #print ("Iteration", i, '\n', self.covariances[0], self.means[0], self.pis[0])
            i+=1
            new_log = self.log_likelihood(data)
            print (new_log)
            if (new_log - old_log) > 0.01 or i < 20:
                old_log = new_log
            else:
                break

    def _initialize_params(self, data):
        km = KMeans(self.k)
        km.fit(data)
        self.dim = data.shape[-1]
        _, self.means = km.predict(data)
        self.means = np.unique(self.means, axis = 0)
        self.pis = np.random.uniform(0,1,(self.k,))
        self.pis = self.pis/np.sum(self.pis)
        self.covariances = np.array([np.eye(self.dim)] * self.k)*100000000
        self.gammas = np.zeros((data.shape[0], self.k))

    def _E_step(self,data):
        #print(self.pis)
        aa = 0
        for i in range(data.shape[0]):
            
            for k in range(self.k):
                a = multivariate_normal(self.means[k], self.covariances[k])
                
                if a.pdf(data[i]) > 0: aa+=1
                self.gammas[i][k] = self.pis[k] * a.pdf(data[i])
        #print(aa/(self.k*data.shape[0]))
        #print(self.gammas)
        self.gammas = self.gammas/np.sum(self.gammas, axis = 1)[None].T
        #print(self.gammas)

    def _M_step(self,data):
        for k in range(self.k):
            self.means[k] = np.dot(self.gammas[:,k].T, data)/np.sum(self.gammas[:,k])
            self.covariances[k] = np.zeros((self.dim, self.dim))
            for i in range(data.shape[0]):
                self.covariances[k] += self.gammas[i][k] * np.dot((data[i]-self.means[k])[None].T, (data[i]-self.means[k][None]))
            self.covariances[k] /= np.sum(self.gammas[:,k])
            self.pis[k] = np.sum(self.gammas[:,k]) / data.shape[0]

    def predict(self, data):
        new_gammas = np.zeros((data.shape[0], self.k))
        for i in range(data.shape[0]):
            for k in range(self.k):
                a = multivariate_normal(self.means[k], self.covariances[k])
                new_gammas[i][k] = self.pis[k] * a.pdf(data[i])
        new_gammas = new_gammas/np.sum(new_gammas, axis = 1)[None].T
        classes = np.argmax(new_gammas, axis = 1)
        return classes

    def get_means(self):
        return self.means.copy()

    def get_covariances(self):
        return self.covariances.copy()

    def get_pis(self):
        return self.pis.copy()
    
    def log_likelihood(self, data):
        return np.sum(np.log(np.sum(self.gammas * self.pis, axis = 1)))
                