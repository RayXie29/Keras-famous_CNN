import keras
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, BatchNormalization, Activation
from keras.layers import Flatten, Input, Dropout, concatenate
from keras.regularizers import  l2
from keras import backend as K
from keras.models import Model


class Inceptionv1_builder():

    def __init__(self, input_shape = (224,224,3), output_units = 1000, init_kernel = (7,7), init_strides = (2,2), init_filters = 64,
                 regularizer = l2(1e-4), initializer = "he_normal", init_maxpooling = True):
        '''
        :param input_shape: input shape of dataset
        :param output_units: output result dimension
        :param init_kernel: The kernel size for first convolution layer
        :param init_strides: The strides for first convolution layer
        :param init_filters: The filter number for first convolution layer
        :param regularizer: regularizer for all the convolution layers in whole NN
        :param initializer: weight/parameters initializer for all convolution & fc layers in whole NN
        :param init_maxpooling: Do the maxpooling after first two convolution layers or not
        '''
        self.input_shape = input_shape
        self.output_units = output_units
        self.init_kernel = init_kernel
        self.init_strides = init_strides
        self.init_filters = init_filters
        self.regularizer = regularizer
        self.initializer = initializer
        self.init_maxpooling = init_maxpooling

        if K.image_dim_ordering() == "tf":

            self.row_axis = 1
            self.col_axis = 2
            self.channel_axis = 3

        else:

            self.row_axis = 2
            self.col_axis = 3
            self.channel_axis = 1


    def _cn_bn_relu(self, filters = 64, kernel_size = (3,3), strides = (1,1), padding = "same",
                    kernel_regularizer = l2(1e-4), kernel_initializer = "he_normal"):
        '''
        convenient function to build convolution -> batch_nromalization -> relu activation layers
        '''
        def f(input_x):

            x = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = padding,
                       kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer)(input_x)
            x = BatchNormalization(axis=self.channel_axis)(x)
            x = Activation("relu")(x)

            return x

        return f

    def _auxiliary(self):
        '''
        In author's explanation:

        " The auxiliary classifier will encourage discrimination in lower stages in the classifier,
        increase the gradient signal that gets propagated back, and provide additional regularization"

        :return: An output layer of auxiliary classifier
        '''
        def f(input_x):

            x = input_x
            x = AveragePooling2D(pool_size=(5,5), strides = (3,3), padding = "same")(x)
            x = self._cn_bn_relu(filters = 128, kernel_size = (1,1), padding = "same",
                                 kernel_regularizer = self.regularizer, kernel_initializer = self.initializer)(x)
            x = Flatten()(x)
            x = Dense(units = 1024)(x)
            x = BatchNormalization(axis = self.channel_axis)(x)
            x = Activation("relu")(x)
            x = Dropout(0.7)(x)

            output = Dense(units = self.output_units, activation = '"softmax', kernel_initializer = self.initializer)(x)
            return output

        return f


    def _inception_block(self, _1x1 = 64, _3x3r = 96, _3x3 = 128, _5x5r = 16, _5x5 = 32, _maxpool = 32):
        '''
        A function for building inception block, including 1x1 convolution layer, 3x3 convolution layer with dimension reducing,
        5x5 convolution layer with dimension reducing and maxpooling layer with dimension reducing

        :param _1x1: filter number for 1x1 convolution layer
        :param _3x3r: filter number for dimension reducing layer of 3x3 convolution layer
        :param _3x3: filter number for 3x3 convolution layer
        :param _5x5r: filter number for dimension reducing layer of 5x5 convolution layer
        :param _5x5: filter number for 5x5 convolution layer
        :param _maxpool: filter number for dimension reducing layer of maxpooling layer
        :return: A concatenate block of several scale convolution which is inception block
        '''
        def f(input_x):

            branch1x1 = self._cn_bn_relu(filters=_1x1, kernel_size=(1, 1), strides=(1, 1), padding="same",
                                         kernel_regularizer= self.regularizer, kernel_initializer=self.initializer)(input_x)

            branch3x3 = self._cn_bn_relu(filters=_3x3r, kernel_size=(1, 1), strides=(1, 1), padding="same",
                                         kernel_regularizer=self.regularizer, kernel_initializer=self.initializer)(input_x)
            branch3x3 = self._cn_bn_relu(filters=_3x3, kernel_size=(3, 3), strides=(1, 1), padding="same",
                                         kernel_regularizer=self.regularizer, kernel_initializer=self.initializer)(branch3x3)

            branch5x5 = self._cn_bn_relu(filters=_5x5r, kernel_size=(1, 1), strides=(1, 1), padding="same",
                                         kernel_regularizer=self.regularizer, kernel_initializer=self.initializer)(input_x)
            branch5x5 = self._cn_bn_relu(filters=_5x5, kernel_size=(5, 5), strides=(1, 1), padding="same",
                                         kernel_regularizer=self.regularizer, kernel_initializer=self.initializer)(branch5x5)

            brancemaxpool = MaxPooling2D(pool_size = (3,3), strides = (1,1), padding = "same")(input_x)
            brancemaxpool = self._cn_bn_relu(filters=_maxpool, kernel_size=(1, 1), strides=(1, 1), padding="same",
                                         kernel_regularizer=self.regularizer, kernel_initializer=self.initializer)(brancemaxpool)

            return concatenate([branch1x1,branch3x3,branch5x5,brancemaxpool], axis = self.channel_axis)

        return f
    def build_inception(self):

        '''
        Main function for building inception nn
        :return: An inception nn
        '''

        #Few traditional convolutional layers at lower layers
        input_x = Input(self.input_shape)
        x = self._cn_bn_relu(filters = self.init_filters, kernel_size = self.init_kernel, strides = self.init_strides,
                             kernel_regularizer = self.regularizer, kernel_initializer = self.initializer)(input_x)

        if self.init_maxpooling:
            x = MaxPooling2D(pool_size = (3,3), strides = (2,2), padding = "same")(x)

        x = self._cn_bn_relu(filters = 64, kernel_size = (1,1), strides = (1,1), padding = "same",
                             kernel_regularizer = self.regularizer, kernel_initializer = self.initializer)(x)
        x = self._cn_bn_relu(filters = 192, kernel_size = (3,3), strides = (1, 1), padding = "same",
                             kernel_regularizer = self.regularizer, kernel_initializer = self.initializer)(x)

        if self.init_maxpooling:
            x = MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = "same")(x)



        #inception(3a)
        x = self._inception_block(_1x1=64, _3x3r=96, _3x3=128, _5x5r=16, _5x5=32, _maxpool=32)(x)

        #inception(3b)
        x = self._inception_block(_1x1=128, _3x3r=128, _3x3=192, _5x5r=32, _5x5=96, _maxpool=64)(x)

        x = MaxPooling2D(pool_size=(3,3), strides = (2,2), padding = "same")(x)

        #inception(4a)
        x = self._inception_block(_1x1=192, _3x3r=96, _3x3=208, _5x5r=16, _5x5=48, _maxpool=64)(x)

        #auxiliary classifier 1
        auxiliary1 = self._auxiliary()(x)

        # inception(4b)
        x = self._inception_block(_1x1=160, _3x3r=112, _3x3=224, _5x5r=24, _5x5=64, _maxpool=64)(x)
        # inception(4c)
        x = self._inception_block(_1x1=128, _3x3r=128, _3x3=256, _5x5r=24, _5x5=64, _maxpool=64)(x)
        # inception(4d)
        x = self._inception_block(_1x1=112, _3x3r=144, _3x3=288, _5x5r=32, _5x5=64, _maxpool=64)(x)

        #auxiliary classifier 2
        auxiliary2 = self._auxiliary()(x)

        # inception(4e)
        x = self._inception_block(_1x1=256, _3x3r=160, _3x3=320, _5x5r=32, _5x5=128, _maxpool=128)(x)

        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding = "same")(x)

        #inception(5a)
        x = self._inception_block(_1x1=256, _3x3r=160, _3x3=320, _5x5r=32, _5x5=128, _maxpool=128)(x)
        #inception(5b)
        x = self._inception_block(_1x1=384, _3x3r=192, _3x3=384, _5x5r=48, _5x5=128, _maxpool=128)(x)

        x_shape = K.int_shape(x)
        x = AveragePooling2D(pool_size = (x_shape[self.row_axis], x_shape[self.col_axis]), strides=(1,1))(x)
        x = Dropout(0.4)(x)
        x = Dense(units = 1000, kernel_initializer = self.initializer)(x)
        x = BatchNormalization(axis = self.channel_axis)(x)
        x = Activation("relu")(x)
        output_x = Dense(units = self.output_units, activation = "softmax", kernel_initializer=self.initializer)(x)

        inceptionv1_model = Model(inputs = [input_x], outputs = [auxiliary1, auxiliary2, output_x])

        return inceptionv1_model


inception_builder = Inceptionv1_builder()
model = inception_builder.build_inception()
model.summary()