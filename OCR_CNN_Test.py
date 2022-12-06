import numpy as np
import cv2
import pickle

width=640
height=480
thersold=65

cap=cv2.VedioCapture(1)
cap.set(3,width)
cap.set(4,height)

pickle_in=open("model_trained.p","rb")
model=pickle.load(pickle_in)

def preprocessing(img):
    img=cv2.cvtcolor(img,cv2.COLOR_BG2GRAY)
    img= cv2.equalizeHist(img)
    img=img/255
    return img

while True:
    success, imgOriginal = cap.read()
    img=np.asarray(imgOriginal)
    img=cv2.resize(img,(32,32))
    img=preprocessing(img)
    cv2.imshow("processed image",img)
    img=img.reshape(1,32,32,1)
    #predict
    classindex=int(model.predict_classes(img))
    #print(classindex)
    prediction=model.predict(img)
    #print(prediction)
    proval=np.amax(prediction)
    print(classindex,proval)

    if proval>thersold:
        cv2.putText(imgOriginal,str(classindex)+" "+str(proval),
                    (50,50),cv2.FONT_HERSHEY_COMPLEX ,1,(0,0,255),1)
    cv2.imshow("original Image",imgOriginal)

    if cv2.waitKey(1) &0xFF ==ord('q'):
       break
