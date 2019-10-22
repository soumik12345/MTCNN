from tensorflow.keras.layers import (
    Input, Conv2D, PReLU,
    MaxPool2D, Reshape,
    Flatten, Dense
)
from tensorflow.keras.models import Model



class Pnet:

    def __init__(self, input_shape = (12, 12, 3)):
        self.input_shape = input_shape
        self.build_network()
    
    
    def pnet_conv_block(self, input_tensor, n_filters, layer_id, pooling = False):
        # Conv Layer
        x = Conv2D(
            n_filters, (3, 3),
            strides = 1, padding = 'valid',
            name = 'pnet_conv_' + str(layer_id)
        )(input_tensor)
        # PReLU Activation Layer
        x = PReLU(
            name = 'pnet_prelu_' + str(layer_id)
        )(x)
        # Pooling Layer
        if pooling:
            x = MaxPool2D(pool_size = 2,
            name = 'pnet_maxpool_' + str(layer_id)
        )(x)
        return x
    

    def face_classifier_block(self, input_tensor):
        x = Conv2D(2, (1, 1), activation = 'softmax', name = 'face_classifier')(input_tensor)
        x = Reshape((2,), name = 'face_classifier_output')(x)
        return x
    

    def bounding_box_prediction_block(self, input_tensor):
        x = Conv2D(4, (1, 1), name = 'bounding_box_regressor')(input_tensor)
        x = Reshape((4,), name = 'bounding_box_regressor_output')(x)
        return x
    

    def facial_landmarks_localizer_block(self, input_tensor):
        x = Conv2D(10, (1, 1), name = 'facial_landmarks_localizer')(input_tensor)
        x = Reshape((10,), name = 'facial_landmarks_localizer_output')(x)
        return x
    
    
    def build_network(self):
        input_layer = Input(shape = self.input_shape)
        x = self.pnet_conv_block(input_layer, 10, 1, pooling = True)
        x = self.pnet_conv_block(x, 16, 2, pooling = False)
        x = self.pnet_conv_block(x, 32, 3, pooling = False)
        face_classification_output = self.face_classifier_block(x)
        bounding_box_predictor = self.bounding_box_prediction_block(x)
        facial_landmarks_localizer = self.facial_landmarks_localizer_block(x)
        self.model = Model(
            input_layer, [
                face_classification_output,
                bounding_box_predictor,
                facial_landmarks_localizer
            ],
            name = 'Proposal_Network'
        )
    

    def summarize(self):
        self.model.summary()




class Rnet:

    def __init__(self, input_shape = (24, 24, 3)):
        self.input_shape = input_shape
        self.build_network()
    

    def rnet_conv_block(self, input_tensor, n_filters, kernel_size, layer_id, pooling = True):
        # Conv Layer
        x = Conv2D(
            n_filters, (kernel_size, kernel_size),
            strides = 1, padding = 'valid',
            name = 'rnet_conv_' + str(layer_id)
        )(input_tensor)
        # PReLU Activation Layer
        x = PReLU(name = 'rnet_prelu_' + str(layer_id))(x)
        # Pooling Layer
        if pooling:
            x = MaxPool2D(
                pool_size = 2, strides = 2,
                name = 'rnet_maxpool_' + str(layer_id)
            )(x)
        return x
    

    def build_network(self):
        input_layer = Input(shape = self.input_shape)
        x = self.rnet_conv_block(input_layer, 28, 3, 1)
        x = self.rnet_conv_block(x, 48, 3, 2)
        x = self.rnet_conv_block(x, 64, 2, 3, pooling = False)
        x = Flatten(name = 'rnet_flatten')(x)
        x = Dense(128, name = 'rnet_fully_connected')(x)
        face_classification_output = Dense(2, activation = 'softmax', name = 'face_classification_output')(x)
        bounding_box_predictor = Dense(4, activation = 'softmax', name = 'bounding_box_output')(x)
        facial_landmarks_localizer = Dense(10, activation = 'softmax', name = 'facial_landmark_output')(x)
        self.model = Model(
            input_layer, [
                face_classification_output,
                bounding_box_predictor,
                facial_landmarks_localizer
            ],
            name = 'Refine_Network'
        )
    

    def summarize(self):
        self.model.summary()



class Onet:

    def __init__(self, input_shape = (48, 48, 3)):
        self.input_shape = input_shape
        self.build_network()
    

    def onet_conv_block(self, input_tensor, n_filters, kernel_size, layer_id, pooling = True):
        # Conv Layer
        x = Conv2D(
            n_filters, (kernel_size, kernel_size),
            strides = 1, padding = 'valid',
            name = 'onet_conv_' + str(layer_id)
        )(input_tensor)
        # PReLU Activation Layer
        x = PReLU(name = 'onet_prelu_' + str(layer_id))(x)
        # Pooling Layer
        if pooling:
            x = MaxPool2D(
                pool_size = 2, strides = 2,
                name = 'onet_maxpool_' + str(layer_id)
            )(x)
        return x

    
    def build_network(self):
        input_layer = Input(shape = self.input_shape)
        x = self.onet_conv_block(input_layer, 32, 3, 1)
        x = self.onet_conv_block(x, 64, 3, 2)
        x = self.onet_conv_block(x, 64, 3, 3)
        x = self.onet_conv_block(x, 128, 2, 4, pooling = False)
        x = Flatten(name = 'onet_flatten')(x)
        x = Dense(256, name = 'onet_fully_connected')(x)
        face_classification_output = Dense(2, activation = 'softmax', name = 'face_classification_output')(x)
        bounding_box_predictor = Dense(4, activation = 'softmax', name = 'bounding_box_output')(x)
        facial_landmarks_localizer = Dense(10, activation = 'softmax', name = 'facial_landmark_output')(x)
        self.model = Model(
            input_layer, [
                face_classification_output,
                bounding_box_predictor,
                facial_landmarks_localizer
            ],
            name = 'Output_Network'
        )
    

    def summarize(self):
        self.model.summary()