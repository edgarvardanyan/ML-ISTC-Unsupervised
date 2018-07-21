#!/usr/bin/env python3
"""
This is a boilerplate file for you to get started on MNIST dataset and run SVD.

This file has code to read labels and data from .gz files you can download from
http://yann.lecun.com/exdb/mnist/

Files will work if train-images-idx3-ubyte.gz file and
train-labels-idx1-ubyte.gz files are in the same directory as this
python file.
"""
from __future__ import print_function
import argparse
import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt
from PCA import PCA
from sklearn.cluster import KMeans

def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mnist-train-data',
                        default='train-images-idx3-ubyte.gz',  # noqa
                        help='Path to train-images-idx3-ubyte.gz file '
                        'downloaded from http://yann.lecun.com/exdb/mnist/')
    parser.add_argument('--mnist-train-labels',
                        default='train-labels-idx1-ubyte.gz',  # noqa
                        help='Path to train-labels-idx1-ubyte.gz file '
                        'downloaded from http://yann.lecun.com/exdb/mnist/')
    args = parser.parse_args(*argument_array)
    return args


def main(args):
    # Read data file into numpy matrices
    with gzip.open(args.mnist_train_data, 'rb') as in_gzip:
        magic, num, rows, columns = struct.unpack('>IIII', in_gzip.read(16))
        all_data = np.array([np.array(struct.unpack('>{}B'.format(rows * columns),
                                           in_gzip.read(rows * columns)))
                    for _ in range(16000)])
    with gzip.open(args.mnist_train_labels, 'rb') as in_gzip:
        magic, num = struct.unpack('>II', in_gzip.read(8))
        all_labels = struct.unpack('>16000B', in_gzip.read(16000))
    each_label = np.empty(10, dtype = object)
    for i in range(10):
        each_label[i] = all_data[np.array(all_labels) == i]
    pca = PCA(15)
    pca.fit(all_data)
    all_data_transform = pca.transform(all_data)
    kmeans_labels = KMeans(n_clusters=10, random_state=0).fit_predict(all_data_transform)
    each_cluster = np.empty(10, dtype = object)
    for i in range(10):
        each_cluster[i] = all_data_transform[:,:2][np.array(kmeans_labels) == i]
    f, axarr = plt.subplots(2, 10, figsize=(18, 4), sharey=True)
    for i in range(10):
        a = pca.transform(each_label[i])
        axarr[0][i].scatter(a.T[0], a.T[1], s = 1)
    for i in range(10):
        axarr[1][i].scatter(each_cluster[i].T[0], each_cluster[i].T[1], s = 1)
    #plt.show()
    coincidence_matrix = np.zeros((10,10)).astype(int)
    for i in range(16000):
        coincidence_matrix[all_labels[i], kmeans_labels[i]]+=1
    print(coincidence_matrix)
    plt.savefig("labels_vs_kmeans_clusters.jpg")

if __name__ == '__main__':
    args = parse_args()
    main(args)