#!/usr/bin/env python3
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from gmm import GMM
import winsound


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--name', help='your name')
    parser.add_argument('--k', type=int, default=15)
    args = parser.parse_args(*argument_array)
    return args


def get_ellipse_from_covariance(matrix, std_multiplier=2):
    values, vectors = np.linalg.eig(matrix)
    maxI = np.argmax(values)
    large, small = values[maxI], values[1 - maxI]
    return (std_multiplier * np.sqrt(large),
            std_multiplier * np.sqrt(small),
            np.rad2deg(np.arccos(vectors[0, 0])))


def main(args):
    df = pd.read_csv(args.data)
    data = np.array(df[['X', 'Y']])
    plt.clf()
    plt.scatter(data[:, 0], data[:, 1], s=3, color='blue')

    gmm = GMM(args.k)
    gmm.fit(data)
    mean = gmm.get_means()
    sigma = gmm.get_covariances()
    pi = gmm.get_pis()

    # Plot ellipses for each of covariance matrices.
    for k in range(len(sigma)):
        #print (sigma[k])
        h, w, angle = get_ellipse_from_covariance(sigma[k])
        e = patches.Ellipse(mean[k], w, h, angle=angle)
        e.set_alpha(np.power(pi[k], .3))
        e.set_facecolor('red')
        plt.axes().add_artist(e)
    plt.savefig('edgar_gmm_sparse.jpg')
    #plt.show()
    winsound.Beep(2500, 1000)


if __name__ == '__main__':
    args = parse_args()
    main(args)
