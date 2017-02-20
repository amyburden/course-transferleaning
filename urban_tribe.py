%matplotlib inline
import os
import glob
import numpy as np
import keras
from keras.applications import VGG16
from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import ModelCheckpoint
import pylab as pl
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy.ma as ma
from mpl_toolkits.axes_grid1 import make_axes_locatable


#Read and prepar data
def data_list():
    folder = './urban_tribe/pictures_all/'
    imgs = glob.glob1(folder, '*.jpg')
    dict = {}
    for img in imgs:
        c = img.split('_')[0]
        if c not in dict:
            dict[c] = list()
        img = os.path.join(folder, img)
        dict.get(c).append(img)
    train = []
    val = []
    test = []
    for c in dict:
        l = np.array(dict[c])
        idx = list(range(len(dict[c])))
        np.random.shuffle(idx)
        train.append(l[idx[: int(len(idx) * 8 / 10)]].tolist())
        val.append(l[idx[int(len(idx) * 8 / 10): int(len(idx) * 9 / 10)]].tolist())
        test.append(l[idx[int(len(idx) * 9 / 10):]].tolist())
    return train, val, test

#read from list
def img_read(train_data, num):
    train_img = []
    train_label = []
    num_of_class = 11
    for i in range(len(train_data)):
        label = np.zeros(num_of_class)
        label[i] = 1
        imagenum = 0
        while imagenum < num and imagenum < len(train_data[i]):
            img = load_img(train_data[i][imagenum], target_size=(224, 224))
            x = img_to_array(img, dim_ordering='tf')
            x = x.reshape((1,) + x.shape)
            train_label.append(label)
            train_img.append(x)
            imagenum += 1
    train_img = np.vstack(train_img)
    train_label = np.vstack(train_label)
    return train_img, train_label


# Create network model
def getModel(output_dim):
    vgg_model = VGG16(weights='imagenet', include_top=True)

    vgg_out = vgg_model.layers[-2].output  # Last FC layer's output
    softmax_layer = Dense(output_dim=output_dim, activation='softmax')(vgg_out)
    # Create new transfer learning model
    tl_model = Model(input=vgg_model.input, output=softmax_layer)

    # Freeze all layers of VGG16 and Compile the model
    for layer in vgg_model.layers:
        layer.trainable = False
    # Confirm the model is appropriate
    tl_model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
    # tl_model.summary()
    return tl_model

# training on urban_tribe data set
def main():
    # Output dim for your dataset
    output_dim = 11  # For Prom2
    train_data, val_data, test_data = data_list()
    for i in [2, 4, 8, 16]:
        tl_model = getModel(output_dim)
        train_img, train_label = img_read(train_data, i)
        train_img = preprocess_input(train_img)

        val_img, val_label = img_read(val_data, 9)
        val_img = preprocess_input(val_img)
        # checkpoint
        os.mkdir('./model')
        checkpointer = ModelCheckpoint(filepath="./model/urban_tribe.hdf5", verbose=1, save_best_only=False)
        # Train the model
        tl_model.fit(train_img, train_label, batch_size=11, nb_epoch=10, validation_data=(val_img, val_label))


#Q4e
def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    pl.colorbar(im, cax=cax)
    pl.savefig('images/4eConvFilter.pdf')


def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]

    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                           dtype=np.float32)

    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = i / ncols
        col = i % ncols

        mosaic[row * paddedh:row * paddedh + imshape[0],
        col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic


def output_layer_img():
    model_path = './model/16/urban_tribe_2_05_1.82.hdf5'
    tl_model = load_model(model_path)
    train_data, val_data, test_data = data_list()
    val_img, val_label = img_read(val_data, 9)
    plt.imshow(val_img[0] / 255.0)
    plt.savefig('images/4eOri.pdf')
    plt.axis('off')
    val_img = preprocess_input(val_img)
    # get intermediate layer
    conv11 = tl_model.get_layer('block1_conv1')
    tmpmodel = Model(input=tl_model.input, output=conv11.output)

    test_img = np.expand_dims(val_img[0], axis=0)
    pred = tmpmodel.predict(test_img)
    pred = np.log(1 + pred.squeeze().transpose(2, 0, 1))  # contrast normalization

    pl.figure(figsize=(15, 15))
    pl.suptitle('convout1_1')
    nice_imshow(pl.gca(), make_mosaic(pred, 8, 8), cmap=cm.binary)

    # get intermediate layer
    conv53 = tl_model.get_layer('block5_conv3')  # -6 is the last conv
    tmpmodel = Model(input=tl_model.input, output=conv53.output)
    pred = tmpmodel.predict(test_img)
    pred = pred.squeeze().transpose(2, 0, 1)  # contrast normalization
    pred = pred[:256]
    pl.figure(figsize=(15, 15))
    pl.suptitle('convout5_3')
    nice_imshow(pl.gca(), make_mosaic(pred, 16, 16), cmap=cm.binary)


# Q5
def getModel5(output_dim):
    vgg_model = VGG16(weights='imagenet', include_top=True)

    vgg_conv53 = vgg_model.get_layer('block4_conv3')  # Last FC layer's output
    vgg_out = Flatten()(vgg_conv53.output)
    softmax_layer = Dense(output_dim=output_dim, activation='softmax')(vgg_out)
    tl_model = Model(input=vgg_model.input, output=softmax_layer)
    for layer in vgg_model.layers:
        layer.trainable = False
    tl_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    tl_model.summary()
    return tl_model


def main5():
    # Output dim for your dataset
    output_dim = 11  # For Prom2
    tl_model = getModel5(output_dim)
    # plot(tl_model, to_file='model.png')
    train_data, val_data, test_data = data_list()
    train_img, train_label = img_read(train_data, 16)
    train_img = preprocess_input(train_img)

    val_img, val_label = img_read(val_data, 9)
    val_img = preprocess_input(val_img)
    tl_model.fit(train_img, train_label, batch_size=11, nb_epoch=7, validation_data=(val_img, val_label))



if __name__ == '__main__':
    # for Q3 and Q4
    main()
    # for Q4e
    # output_layer_img()
    # for Q5
    # main5()