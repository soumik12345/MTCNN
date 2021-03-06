from tensorflow.keras.layers import (
    Input, Conv2D, Activation,
    MaxPool2D, Dense, Flatten
)
from tensorflow.keras.models import Model


class Net12:

    def __init__(self, input_shape = (12, 12, 3)):
        self.input_shape = input_shape
        self.build_network()
    

    def build_network(self):
        input_layer = Input(shape = self.input_shape)
        x = Conv2D(16, (3, 3), strides = 1, padding = 'same')(input_layer)
        x = MaxPool2D(pool_size = 3, strides = 2)(x)
        x = Activation('relu')(x)
        x = Flatten()(x)
        x = Dense(16)(x)
        x = Activation('relu')(x)
        x = Dense(2, activation = 'softmax')(x)
        self.model = Model(input_layer, x)
    

    def summarize(self):
        self.model.summary()



class Net12Calibration:

    def __init__(self, input_shape = (12, 12, 3)):
        self.input_shape = input_shape
        self.build_network()
    

    def build_network(self):
        input_layer = Input(shape = self.input_shape)
        x = Conv2D(16, (3, 3), strides = 1, padding = 'same')(input_layer)
        x = MaxPool2D(pool_size = 3, strides = 2)(x)
        x = Activation('relu')(x)
        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation('relu')(x)
        x = Dense(45, activation = 'softmax')(x)
        self.model = Model(input_layer, x)
    

    def summarize(self):
        self.model.summary()