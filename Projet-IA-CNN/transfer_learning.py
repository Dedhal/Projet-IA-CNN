from tensorflow import keras
from skimage import io, transform
import cv2
import pickle
import numpy as np

model = keras.models.load_model("G:\\Datasets\\CoCo\\models\\baby_cnn_9.h5")
model.summary()
DIM = 224
SCATTERED = "G:\\Datasets\\CoCo\\scattered\\"

def process_image(img):
    X = np.empty((1, *(DIM,DIM), 3))
    means = None
    with open('means_scatter.bin', 'rb') as f:
            means = pickle.load(f)
    downsample_size = DIM
    img_read = io.imread(img)
   
    try:
        img_read = cv2.cvtColor(img_read,cv2.COLOR_GRAY2RGB)
   
    except:
        pass
   
    img_read = transform.resize(img_read, (downsample_size, downsample_size), mode='constant')
    img_read = (img_read - means)
   
    X[0,] = img_read
    print(X.shape)
    return X

#def one_hot_converter(liste):
#    res = []
#    for i in [5, 19, 32, 52, 90]:
#        if i in liste:
#            res.append(1)
#        else:
#            res.append(0)
    
#    return res


#x = process_image(SCATTERED + "test\\data\\" + "000000012032.jpg")
##y = [1, 0, 1, 0, 0]
#print(model.predict(x))

#x = process_image(SCATTERED + "test\\data\\" + "000000025830.jpg")
#print(model.predict(x))

#x = process_image(SCATTERED + "train\\data\\" + "000000000049.jpg")
#print(model.predict(x))

#x = process_image(SCATTERED + "train\\data\\" + "000000000247.jpg")
#print(model.predict(x))

print(model.predict(process_image("G:\\Datasets\\stop-sign.jpg")))