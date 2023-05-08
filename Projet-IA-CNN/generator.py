import numpy as np
import keras
from skimage import io, transform
from skimage.util import crop
import matplotlib.pyplot as plt
import cv2
import pickle

DIM = 224
BATCH_SIZE = 8

FULL = "G:\\Datasets\\CoCo\\"
REDUCED = "G:\\Datasets\\CoCo\\reduced\\"
SCATTERED = "G:\\Datasets\\CoCo\\scattered\\"
CIFAR100 = "G:\\Datasets\\Cifar100\\"

MEAN_REDUCED = "means_reduced.bin"
MEAN_SCATTERED = "means_scatter.bin"
MEAN_CIFAR100 = "means_cifar.bin"

USE_DATASET = SCATTERED

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

    def process_image(self, img, datas=None, category=None):
        downsample_size = DIM
        img_read = io.imread(img)


        def search_for_object(datas, category):
            objects = None
            if datas != None:
                for i, j in zip(datas, category):
                    if j in [5, 19, 32, 52, 90]:
                        objects = img_read[int(i[1]):int(i[1] + i[3]), int(i[0]):int(i[0] + i[2])]
                        return objects

            return None

        objects = None
        objects = search_for_object(datas, category)

        if type(objects) == type(None) and type(datas) != type(None):
            return None, None

        elif type(objects) != type(None) and type(datas) != type(None):
            try:
                objects = cv2.cvtColor(objects,cv2.COLOR_GRAY2RGB)

            except:
                pass

            y = self.one_hot_converter_scatter(category)

            #io.imshow(objects)
            #plt.show()
            try:
                objects = transform.resize(objects, (downsample_size, downsample_size), mode='constant')
                #objects = (objects - self.means)
                return objects, y
            except:
                return None, None

        else:
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
        Y = np.empty((self.batch_size, 5),dtype=int)

        j = 0
        for i, img in enumerate(list_IDs_temp):
            if self.mode == "train":
                if USE_DATASET == CIFAR100:
                    x = self.process_image(USE_DATASET + "train\\data\\" + str(img) + ".jpg")
                else:
                    x, y = self.process_image(USE_DATASET + "train\\data\\" + str("%012d" % (img,)) + ".jpg", 
                                           self.datas[self.datas["image_id"] == img]["bbox"].to_list(), 
                                           self.datas[self.datas["image_id"] == img]["category_id"].to_list())
                    if type(x) == type(None):
                        continue
            else:
                if USE_DATASET == CIFAR100:
                    x = self.process_image(USE_DATASET + "test\\data\\" + str(img) + ".jpg")
                else:
                    x, y = self.process_image(USE_DATASET + "validation\\data\\" + str("%012d" % (img,)) + ".jpg", 
                                           self.datas[self.datas["image_id"] == img]["bbox"].to_list(), 
                                           self.datas[self.datas["image_id"] == img]["category_id"].to_list())
                    if type(x) == type(None):
                        continue
            #y = self.one_hot_converter_scatter(list(set(self.datas[self.datas["image_id"] == img]["category_id"].to_list())))
            #y = np.array(y).reshape(1, 5)

            X[j,] = x
            Y[j] = y

            j += 1

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
