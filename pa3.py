
import vgg16_starter
from vgg16_starter import getModel
from data_list import data_list
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from matplotlib import pyplot as plt
import numpy as np


model = getModel(256)

train_data, val_data, test_data = data_list()
train_img = []
train_label = []
classnum = 0
while classnum < 256:
    label = np.zeros(256)
    label[classnum] = 1
    imagenum = 0
    while imagenum < 2:
        img = load_img(train_data[classnum][imagenum],target_size = (224,224))
        x = (img_to_array(img,dim_ordering = 'th'))
        x = x.reshape((1,) + x.shape)[:, ::-1, :, :]
        train_label.append(label)
        train_img.append(x)
        imagenum += 1
    classnum += 1
train_img = np.vstack(train_img)
train_label = np.vstack(train_label)

calTech256 = ImageDataGenerator(samplewise_center= True,
                                zca_whitening=True,
                                samplewise_std_normalization=True,
                                rescale = 1./255)


calTech256.fit(train_img)
training = calTech256.flow(train_img,train_label)



# In[ ]:

sgd = SGD(lr=1e-3, decay=1e-4, momentum=0.9, nesterov=True)
model.compile(optimizer = sgd, loss = 'categorical_crossentropy')
his = model.fit_generator(training,512,10)


# In[ ]:

plabel = np.argmax(model.predict(train_img),axis = 1)
print  np.mean(train_label[np.arange(train_label.shape[0],plabel)])


# In[ ]:

print  np.mean(train_label[np.arange(train_label.shape[0]),plabel])


# In[ ]:

print 1/np.mean(train_label[np.arange(train_label.shape[0]),plabel])


# In[ ]:
