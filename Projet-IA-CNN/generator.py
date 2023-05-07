import numpy as np
import keras
from skimage import io, transform
import cv2
import pickle

DIM = 224
BATCH_SIZE = 32

FULL = "G:\\Datasets\\CoCo\\"
REDUCED = "G:\\Datasets\\CoCo\\reduced\\"
SCATTERED = "G:\\Datasets\\CoCo\\scattered\\"
CIFAR100 = "G:\\Datasets\\Cifar100\\"

MEAN_REDUCED = "means_reduced.bin"
MEAN_SCATTERED = "means_scatter.bin"
MEAN_CIFAR100 = "means_cifar.bin"

USE_DATASET = CIFAR100

if USE_DATASET == REDUCED:
    MEAN_FILE = MEAN_REDUCED
elif USE_DATASET == SCATTERED:
    MEAN_FILE = MEAN_SCATTERED
elif USE_DATASET == CIFAR100:
    MEAN_FILE = MEAN_CIFAR100

class DataGenerator(keras.utils.Sequence):

    def __init__(self, datas, mode="train", batch_size=BATCH_SIZE, dim=(DIM,DIM), n_channels=3, n_classes=5, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.datas = datas
        self.mode = mode
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        with open(MEAN_FILE, 'rb') as f:
            self.means = pickle.load(f)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.datas["image_id"].unique()))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def one_hot_cifar100(self, liste):
        res = []
        for i in range(100):
            if i in liste:
                res.append(1)
            else:
                res.append(0)
    
        return res
    

    def one_hot_converter_scatter(self, liste):
        res = []
        for i in [5, 19, 32, 52, 90]:
            if i in liste:
                res.append(1)
            else:
                res.append(0)
    
        return res

    def one_hot_converter(self, liste):
        res = []
        for i in range(52, 57):
            if i in liste:
                res.append(1)
            else:
                res.append(0)
    
        return res

    def process_image(self, img):
        downsample_size = DIM
        img_read = io.imread(img)

        try:
            img_read = cv2.cvtColor(img_read,cv2.COLOR_GRAY2RGB)

        except:
            pass

        img_read = transform.resize(img_read, (downsample_size, downsample_size), mode='constant')
        img_read = (img_read - self.means)
        
        return img_read

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, 100),dtype=int)

        for i, img in enumerate(list_IDs_temp):
            if self.mode == "train":
                if USE_DATASET == CIFAR100:
                    x = self.process_image(USE_DATASET + "train\\data\\" + str(img) + ".jpg")
                else:
                    x = self.process_image(USE_DATASET + "train\\data\\" + str("%012d" % (img,)) + ".jpg")
            else:
                if USE_DATASET == CIFAR100:
                    x = self.process_image(USE_DATASET + "test\\data\\" + str(img) + ".jpg")
                else:
                    x = self.process_image(USE_DATASET + "validation\\data\\" + str("%012d" % (img,)) + ".jpg")
            y = self.one_hot_cifar100(list(set(self.datas[self.datas["image_id"] == img]["category_id"].to_list())))
            #y = np.array(y).reshape(1, 5)

            X[i,] = x
            Y[i] = y

        return X, Y  #keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.datas["image_id"].unique()) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.datas["image_id"].unique()[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
