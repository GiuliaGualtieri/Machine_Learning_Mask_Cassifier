# import the libraries
import cv2
import numpy as np
import pickle
import socket, pickle
from PIL import Image

with open("models/model.pkl", 'rb') as f:
  model_mask = pickle.load(f)

# define the target class
class_names = ["no-mask", "mask"]

filepath = "data/test_image_2.jpg"
im = cv2.imread(filepath, cv2.IMREAD_COLOR)

# La colormap e BGR convertila a RGB per visualizzazione correta
im =cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# Resize Immagine
im = cv2.resize(im, (64,64))

img = Image.open('data/test_image_2.jpg')
img.show()

im=im.reshape(1,-1)
#im.shape

y_hat=model_mask.predict_proba(im)
#y_hat

idx = np.argmax(y_hat)


cv2.waitKey(300)
#print(idx)
print(class_names[idx])

