import keras
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Flatten, Dense


def getModel( output_dim ):
    '''
        * output_dim: the number of classes (int)

        * return: compiled model (keras.engine.training.Model)
    '''
    vgg_model = VGG16( weights='imagenet', include_top=True )
    vgg_out = vgg_model.layers[-2].output #Last FC layer's output
    cv5_out = vgg_model.layers[-6].output
    cv4_out = vgg_model.layers[-10].output

    x5 = Flatten(name='flatten5')(cv5_out)
    x4 = Flatten(name='flatten4')(cv4_out)

    for layer in vgg_model.layers:
        layer.trainable = False
    softmax_layer = Dense(output_dim , init = 'normal', activation = 'softmax')(vgg_out)     #Create softmax layer taking input as vgg_out
    softmax_5 = Dense(output_dim , init = 'normal', activation = 'softmax')(x5)
    softmax_4 = Dense(output_dim , init = 'normal', activation = 'softmax')(x4)

    tl_model = Model(input=vgg_model.input, output=softmax_layer)
    tl_model5 = Model(input=vgg_model.input, output=softmax_5)
    tl_model4 = Model(input=vgg_model.input, output=softmax_4)

    #Freeze all layers of VGG16 and Compile the model
    #Confirm the model is appropriate

    return tl_model, tl_model5, tl_model4

if __name__ == '__main__':
    #Output dim for your dataset
    output_dim = 256 #For Caltech256

    tl_model = getModel( output_dim )
    #Train the model
    #Test the model
