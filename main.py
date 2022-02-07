# import the libraries
import cv2
import numpy as np
import pickle
import socket
import joblib
from PIL import Image

# load the model
#with open("models/modelv1.pkl", 'rb') as f:
  #model_mask = pickle.load(f)
model_mask = joblib.load("models/modelv.1.1.0.pkl")

# define the target class
class_names =  ["mask", "no-mask"]

filepath = "data/test_image_1.jpg"
img = Image.open('data/test_image_1.jpg')
img.show()
im = cv2.imread(filepath, cv2.IMREAD_COLOR)

# La colormap e BGR convertila a RGB per visualizzazione correta
im =cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# Resize Immagine
im = cv2.resize(im, (64,64))

im=im.reshape(1,-1)
#im.shape

y_hat=model_mask.predict_proba(im)
#y_hat

#if y_hat[0,1]>0.10:
#  idx=1
#else:
#  idx=0

idx = np.argmax(y_hat)
#print(y_hat.shape) 

cv2.waitKey(300)
#print(idx)
print('Prob of having a mask: ', y_hat[0,0])
print('Prob of not having a mask: ', y_hat[0,1])
print('answer is: ', class_names[idx])

