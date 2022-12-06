import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras import Sequential
from keras import Conv2D
import pickle


path='my_data'
test_Ratio=0.2
valRatio=0.2
imageDimensions=(32,32,3)
images=[]
classNo=[]
mylist=os.listdir(path)
print("total no of class detected ",len(mylist))

noOfclasses=len(mylist)

print("importing classes ...")
for x in range(0,noOfclasses):
    myPicList =os.listdir(path+"/"+str(x))
    for y in myPicList:
        curImg =cv2.imread(path+"/"+str(x)+"/"+y)
        curimg=cv2.resize(curImg,(imageDimensions[0],imageDimensions[1]))
        images.append(curImg)
        classNo.append(x)
    print(x,end=" ")
print(" ")

images=np.array(images)
classNo=np.array(classNo)

print(images.shape)
#print(classNo.shape)


#spltiing data

### Spliting the data

x_train,x_test,y_train,y_test=train_test_split(images,classNo,test_size=test_Ratio)
x_train,x_validation,y_train,y_validation = train_test_split(x_train,y_train,test_size=valRatio)



#print(x_train.shape)
#print(x_test.shape)
#print(x_validation.shape)

#print(np.where(y_train==0))
noOfSamples=[]
for x in range(0,noOfclasses):
   print(np.where(y_train==0)[0])
   noOfSamples.append(len(np.where(y_train==x)[0]))

print(noOfSamples)

plt.figure(figsize=(10,5))
plt.bar(range(0,noOfclasses),noOfSamples)

plt.title("no of Images for each class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images")
plt.show()
def preprocessing(img):
    img=cv2.cvtcolor(img,cv2.COLOR_BG2GRAY)
    img= cv2.equalizeHist(img)
    img=img/255
    return img

# img=preprocessing(x_train[30])
# img=cv2.resize(img,(300,300))
# cv2.imshow("PreProcessed",img)
# cv2.waitkey(0)

x_train=np.array(list(map(preprocessing,x_train)))

# img=(x_train[30])
# img=cv2.resize(img,(300,300))
# cv2.imshow("PreProcessed",img)
# cv2.waitkey(0)
x_test=np.array(list(map(preprocessing,x_test)))
x_validtion=np.array(list(map(preprocessing,x_validation)))

#print(x_train.shape),before reshape
x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2])
#print(x_train.shape),after reshape

x_test=x_train.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2])
x_validation=x_train.reshape(x_validation.shape[0],x_validation.shape[1],x_validation.shape[2])


dataGen =ImageDataGenerator(width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.2,
                            shear_range=0.1,
                            rotation_range=10)

dataGen.fit(x_train)
#one hot encoding
y_train=to_categorical(y_train,noOfclasses)
y_test=to_categorical(y_test,noOfclasses)
y_validation=to_categorical(y_validation,noOfclasses)

def myModel():
    noOFfilter=60
    sizeoFfilter1=(5,5)
    sizeoFfilter2=(3,3)
    sizeOfPool=(2,2)
    noOfNode=500

    model = Sequential()
    model.add((Conv2D(noOFfilter,sizeoFfilter1,input_shape=(imageDimensions[0],
                                                            imageDimensions[1],
                                                            1),activation='relu')))

    model.add((Conv2D(noOFfilter, sizeoFfilter1,  activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOFfilter//2,sizeoFfilter2,activation='relu')))
    model.add((Conv2D(noOFfilter // 2, sizeoFfilter2, activation='relu')))
    model.add(Dropout(0.5))

    model.add(Flatten())#flattening layer

    #dense layer
    #define no of node
    model.add(Dense(noOfNode,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfNode, activation='softmax'))
    model.compile(Adam(lr=0.001),loss='categorical crossentropy',
                  metrices=['accuracy'])
    return model
model=myModel()
print(model.summary)

batchSizeVal=50
epochVal=10
stepsPerEpoch=2000

history=model.fit_generator(dataGen.flow(x_train,y_train,
                                 batch_size=batchSizeVal),
                                    steps_per_epoch=stepsPerEpoch,
                                        epochs=epochVal,
                                          validation_data=(x_validation,y_validation),
                                            shuffle=1
                    )
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['Accuracy'])
plt.plot(history.history['val_Accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()
score=model.evaluate(x_test,y_test,verbose=0)
print('Test Score= ',score[0])
print('Test Accuracy= ',score[1])

pickle_out=open("model_train.p","wb")
pickle.dump(model,pickle_out)
pickle_out.close()

 






