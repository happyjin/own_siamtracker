# from PIL import Image
import sys
import os
import math
import numpy as np
import deepdish as dd
import torch.utils.data as data
import matplotlib.pyplot as plt
from scipy.misc import imsave

###########################################################################################
# script to generate moving mnist video dataset (frame by frame) as described in
# [1] arXiv:1502.04681 - Unsupervised Learning of Video Representations Using LSTMs
#     Srivastava et al
# adopted from Tencia Lee
###########################################################################################

# helper functions
def arr_from_img(im,shift=0):
    w,h=im.size
    arr=im.getdata()
    c = np.product(arr.size) / (w*h)
    return np.asarray(arr, dtype=np.float32).reshape((h,w,c)).transpose(2,1,0) / 255. - shift

def get_picture_array(X, index, shift=0):
    ch, w, h = X.shape[1], X.shape[2], X.shape[3]
    ret = ((X[index]+shift)*255.).reshape(ch,w,h).transpose(2,1,0).clip(0,255).astype(np.uint8)
    if ch == 1:
        ret=ret.reshape(h,w)
    return ret

# loads mnist from web on demand
def load_dataset():
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve
    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)
    import gzip
    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28).transpose(0,1,3,2)
        return data / np.float32(255)
    return load_mnist_images('train-images-idx3-ubyte.gz')

class MovingMnist(data.Dataset):
    def __init__(self, seq_len=20, nums_per_image=2, fake_dataset_size=1000):
        self.epoch_length = fake_dataset_size
        self.seq_len = seq_len
        self.mnist = load_dataset()
        self.nums_per_image = nums_per_image
        self.img_size = 80
        self.digit_size = 28
        self.img_shape = (self.img_size, self.img_size)
        self.digit_shape = (self.digit_size, self.digit_size)
        self.x_lim = self.img_shape[0]-self.digit_shape[0]
        self.y_lim = self.img_shape[1]-self.digit_shape[1]
        self.lims = (self.x_lim, self.y_lim)

    def __getitem__(self, index):
        # randomly generate direc/speed/position, calculate velocity vector
        top_left_coords = []
        direcs = np.pi * (np.random.rand(self.nums_per_image)*2 - 1)
        speeds = np.random.randint(5, size=self.nums_per_image)+2
        veloc = [(v*math.cos(d), v*math.sin(d)) for d,v in zip(direcs, speeds)]
        mnist_images = [self.mnist[r] for r in np.random.randint(0, self.mnist.shape[0], self.nums_per_image)]
        positions = [(np.random.rand()*self.x_lim, np.random.rand()*self.y_lim) for _ in range(self.nums_per_image)]

        data = np.zeros((self.seq_len, self.img_size, self.img_size))
        for frame_idx in range(self.seq_len):
            top_left_coords.append(positions)
            for i, digit in enumerate(mnist_images):
                x,y = int(positions[i][0]), int(positions[i][1])
                data[frame_idx, x:x+self.digit_shape[0], y:y+self.digit_shape[1]] += digit[0]
            # update positions based on velocity
            next_pos = [list(map(sum, zip(p,v))) for p,v in zip(positions, veloc)]
            #print(next_pos)
            # bounce off wall if a we hit one
            for i, pos in enumerate(next_pos):
                for j, coord in enumerate(pos):
                    if coord < 0 or coord > self.lims[j]:
                        veloc[i] = tuple(list(veloc[i][:j]) + [-1 * veloc[i][j]] + list(veloc[i][j+1:]))
            positions = [list(map(sum, zip(p,v))) for p,v in zip(positions, veloc)]

        top_left_coords = np.array(top_left_coords)

        return np.clip(data, 0, 1), top_left_coords
        #return data, centroids

    def __len__(self):
        return self.epoch_length

if __name__ == '__main__':
    n_data = 20
    seq_len = 100
    dataset = []
    top_lefts = []
    for i in range(n_data):
        ds = MovingMnist(seq_len=seq_len)
        sub_dataset, top_left = ds.__getitem__(i)
        dataset.append(sub_dataset)
        top_lefts.append(top_left)

    dataset = np.array(dataset)
    top_lefts = np.array(top_lefts)
    print(dataset.shape) #(n_subdataset, n_frames, height, width)
    print(top_lefts.shape) #(n_sub, n_frames, y, x) for two digits with two coordinates
    for i in range(n_data):
        directory = os.path.join('../DATASET/Moving_MNIST/', str(i))
        if not os.path.exists(directory):
            os.makedirs(directory)
        for j in range(seq_len):
            stacked_img = np.stack((dataset[i][j],) * 3, axis=-1)
            imsave(os.path.join(directory, str(j)+'.jpg'), stacked_img)


    # save information
    data_name = '../DATASET/Moving_MNIST/digits_normal.h5'
    coord_name = '../DATASET/Moving_MNIST/top_lefts_normal.h5'
    dd.io.save(data_name, dataset)
    dd.io.save(coord_name, top_lefts)


'''
    ds = MovingMnist(seq_len=50)
    #print(ds[0])
    results, centroids = ds.__getitem__(0)
    columns = 4
    rows = 4
    fig = plt.figure(figsize=(columns, rows))
    for i in range(1, columns * rows + 1):
        print(i)
        print(results[i-1].shape)
        #img = np.random.randint(10, size=(h, w))
        fig.add_subplot(columns, rows, i)
        #plt.plot([centroids[i, 0]+14], [centroids[i,1]+14], marker='o', markersize=3, color="red")
        plt.plot([centroids[i-1][:,1]], [centroids[i-1][:,0]], marker='o', markersize=3, color="red")
        plt.imshow(results[i-1])
    plt.show() 
'''





