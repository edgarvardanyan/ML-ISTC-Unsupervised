from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


if __name__ == '__main__':
    images = np.array([mpimg.imread(f'gen_img{i}.png').flatten() for i in range(5)])
    print (images.shape)
    ica = FastICA(3, whiten = True)
    a = ica.fit_transform(images.T).T

    signs = np.array([-1,-1,1])[None].T
    new_images = (signs * a * (images.std()/a.std()) + images.mean()).clip(0,1).reshape(3,800,600,4)
    for i in range(3):
        plt.imshow(new_images[i] - new_images[i].min())
        plt.imsave(f'source_img{i}.png', new_images[i])
        plt.show()