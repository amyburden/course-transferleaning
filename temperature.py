
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Input, Dense
from keras import backend as K


def getTempModel(output_dim,T=1):
    def temp_softmax(x):
        return K.softmax(x/T)
    vgg_model = VGG16(weights='imagenet', include_top=True)

    vgg_out = vgg_model.layers[-1].output  # Last FC layer's output

    softmax_layer = Dense(output_dim=output_dim, activation=temp_softmax)(vgg_out)
    # Create new transfer learning model
    tl_model = Model(input=vgg_model.input, output=softmax_layer)

    # Freeze all layers of VGG16 and Compile the model
    vgg_model.trainable = False
    for layer in vgg_model.layers:
        layer.trainable = False
    # Confirm the model is appropriate

    return tl_model