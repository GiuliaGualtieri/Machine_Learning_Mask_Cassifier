# 0.0) import the libraries
import cv2
import numpy as np
import pickle 

# 0.1) definition of supporting functions
def rotate(image, angle, center=None, scale=1.0):
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # return the rotated image
    return rotated

def  findLargestBB(bbs):
  areas = [w*h for x,y,w,h in bbs]
  if not areas:
      return False, None
  else:
      i_biggest = np.argmax(areas) 
      biggest = bbs[i_biggest]
      return True, biggest

# 1.0) capture the photo througth VideoCamera on your PC
cap = cv2.VideoCapture(0) 
# if videocamera is not available you can upload a video as follows:
# cv2.VideoCapture("data/video.mp4")

if (cap.isOpened()== False): 
  print("Error opening video stream or file")


# 2.0) load the model imported from library cv in order to be able to detect
# different faces into a frame.
model_face = cv2.CascadeClassifier('models/haar-cascade-files/haarcascade_frontalface_default.xml')

# Load mask classifier
#with open("models/mask-classifiers/model.pkl", 'rb') as f:
#  model_mask = pickle.load(f)

# 3.0) load the pickle format model 
with open("models/model.pkl", 'rb') as f:
  model_mask = pickle.load(f)

# 4.0) define the target class
class_names = ["mask", "no-mask"]

# 5.0) estimate the presence of mask untill the video go down
while(cap.isOpened()):

  # lettura immagine
  ret, frame = cap.read()  #BGR
  
  # coversion image from BGR to RGB
  #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

  # Rotation of the frame
  #frame = rotate(frame, -90)

  # Find all images in the frame
  faces = model_face.detectMultiScale(frame,scaleFactor=1.4,minNeighbors=4, flags=cv2.CASCADE_DO_ROUGH_SEARCH | cv2.CASCADE_SCALE_IMAGE)
  
  # Find the largest face in the frame
  retFace, facebig = findLargestBB(faces)

  
  # For each face
  if retFace == True:

    # Extra coordiante of largest image
    x,y,w,h = facebig
    
    # Crop image 
    roi = frame[y:y+h,x:x+w]

    # -----
    # 64x64x3 --> put the frame on a line/vector of shape 1x12288
    roi = cv2.resize(roi,(64,64))
    inpX = roi.reshape(1,-1)
    y_hat = model_mask.predict_proba(inpX)
    idx = np.argmax(y_hat)
    cv2.putText(frame, class_names[idx] + " " +  str(np.round(y_hat[0][idx])), (50, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
   
    # print(y_hat)
    # print(int(y_hat))
    # y_hat = model_mask.predict(inpX)
    # cv2.putText(frame, class_names[int(y_hat[0])], (50, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
  
    # print(y_hat)
    # -----
    

    # Drow the rectangle around the largest face
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    # Only on your PC
    #cv2.imshow("Roi", roi)

  # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
  cv2.imshow("Image", frame)
  # waits for a key event for a "delay" (here, 30 milliseconds)
  cv2.waitKey(33) 

  # In colab o jupyter
  #frame = cv2.resize(frame, (128,128))
  #im_pil = Image.fromarray(roi)
  #display(im_pil)