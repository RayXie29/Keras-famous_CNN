import keras
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, BatchNormalization
from keras.layers import Flatten, Input, Dropout, concatenate
from keras.regularizers import  l2
from keras import backend as K
from keras.models import Model


class Inceptionv2_builder():

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

        assert len(input_shape) == 3, "input shape should be dim 3 ( row, col, channel or channel row col )"

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
        convenient function to build convolution -> batch normalization -> relu activation layers
        '''
        def f(input_x):

            x = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = padding,
                       kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer)(input_x)
            x = BatchNormalization(axis = self.channel_axis)(x)
            x = Activation("relu")(x)

            return x

        return f

    def _auxiliary(self, name = "auxiliary_1"):
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
            x = BatchNormalization(axis = 1)(x)
            x = Activation("relu")(x)
            x = Dropout(0.7)(x)

            output = Dense(units = self.output_units, activation = "softmax", kernel_initializer = self.initializer, name = name)(x)
            return output

        return f


    def _inception_block(self, _1x1 = 64, _3x3r = 96, _3x3 = 128, _d3x3r = 16, _d3x3 = 32,
                         _pool = 32, strides = (1,1), pooling = "avg", name = "inception3a"):
        '''
        A function for building inception block, including 1x1 convolution layer, 3x3 convolution layer with dimension reducing,
        double 3x3 convolution layers with dimension reducing and maxpooling layer with dimension reducing

        :param _1x1: filter number of 1x1 convolution layer
        :param _3x3r: filter number of dimension reducing layer for 3x3 convolution layer
        :param _3x3: filter number of 3x3 convolution layer
        :param _d3x3r: filter number of dimension reducing layer for double 3x3 convolution layers
        :param _d3x3: filter number of double 3x3 convolution layers
        :param _maxpool: filter number of dimension reducing layer for maxpooling layer
        :return: A concatenate block of several scale convolution which is inception block
        '''
        def f(input_x):

            branch1x1 = None
            if _1x1 > 0:
                branch1x1 = self._cn_bn_relu(filters=_1x1, kernel_size=(1, 1), strides=(1, 1), padding="same",
                                             kernel_regularizer=self.regularizer, kernel_initializer=self.initializer)(input_x)

            branch3x3 = self._cn_bn_relu(filters=_3x3r, kernel_size=(1, 1), strides=(1, 1), padding="same",
                                         kernel_regularizer=self.regularizer, kernel_initializer=self.initializer)(input_x)
            branch3x3 = self._cn_bn_relu(filters=_3x3, kernel_size=(3, 3), strides= strides, padding="same",
                                         kernel_regularizer=self.regularizer, kernel_initializer=self.initializer)(branch3x3)

            dbranch3x3 = self._cn_bn_relu(filters=_d3x3r, kernel_size=(1, 1), strides = (1, 1), padding="same",
                                         kernel_regularizer=self.regularizer, kernel_initializer=self.initializer)(input_x)
            dbranch3x3 = self._cn_bn_relu(filters=_d3x3, kernel_size=(3, 3), strides = (1, 1), padding="same",
                                         kernel_regularizer=self.regularizer, kernel_initializer=self.initializer)(dbranch3x3)
            dbranch3x3 = self._cn_bn_relu(filters=_d3x3, kernel_size=(3, 3), strides = strides, padding="same",
                                          kernel_regularizer=self.regularizer, kernel_initializer=self.initializer)(dbranch3x3)

            brancemaxpool = None
            if pooling == "avg":
                brancemaxpool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(input_x)
                brancemaxpool = self._cn_bn_relu(filters=_pool, kernel_size=(1, 1), strides=(1, 1), padding="same",
                                                 kernel_regularizer=self.regularizer, kernel_initializer=self.initializer)(brancemaxpool)
            else:
                brancemaxpool = MaxPooling2D(pool_size=(3,3), strides = strides, padding = "same")(input_x)

            if _1x1 > 0:
                return concatenate([branch1x1, branch3x3, dbranch3x3, brancemaxpool], axis=self.channel_axis, name = name)
            else :
                return concatenate([branch3x3, dbranch3x3, brancemaxpool], axis=self.channel_axis, name = name)

        return f

    def build_inception(self):

        '''
        Main function for building inceptionV2 nn
        :return: An inceptionV2 nn
        '''

        #Few traditional convolutional layers at lower layers
        input_x = Input(self.input_shape)
        x = self._cn_bn_relu(filters = self.init_filters, kernel_size = self.init_kernel, strides = self.init_strides,
                             kernel_regularizer = self.regularizer, kernel_initializer = self.initializer)(input_x)

        if self.init_maxpooling:
            x = MaxPooling2D(pool_size = (3,3), strides = (2,2), padding = "same")(x)

        x = self._cn_bn_relu(filters = 192, kernel_size = (3,3), strides = (1, 1), padding = "same",
                             kernel_regularizer = self.regularizer, kernel_initializer = self.initializer)(x)

        if self.init_maxpooling:
            x = MaxPooling2D(pool_size = (3,3), strides = (2,2), padding = "same")(x)



        #inception(3a)
        x = self._inception_block(_1x1=64, _3x3r=64, _3x3=64, _d3x3r=64, _d3x3=96, _pool=32, name = "inception3a")(x)

        #inception(3b)
        x = self._inception_block(_1x1=64, _3x3r=64, _3x3=96, _d3x3r=64, _d3x3=96, _pool=64, name = "inception3b")(x)

        #inception(3c)
        x = self._inception_block(_1x1=0, _3x3r=128, _3x3=160, _d3x3r=64, _d3x3=96,_pool=0,
                                  name = "inception3c", strides=(2,2),pooling="max")(x)

        #inception(4a)
        x = self._inception_block(_1x1=224, _3x3r=64, _3x3=96, _d3x3r=96, _d3x3=128, _pool=128, name = "inception4a")(x)

        #auxiliary classifier 1
        auxiliary1 = self._auxiliary(name = "auxiliary_1")(x)

        # inception(4b)
        x = self._inception_block(_1x1=192, _3x3r=96, _3x3=128, _d3x3r=96, _d3x3=128, _pool=128, name = "inception4b")(x)
        # inception(4c)
        x = self._inception_block(_1x1=160, _3x3r=96, _3x3=128, _d3x3r=128, _d3x3=160, _pool=128, name = "inception4c")(x)
        # inception(4d)
        x = self._inception_block(_1x1=96, _3x3r=128, _3x3=160, _d3x3r=160, _d3x3=192, _pool=128, name = "inception4d")(x)

        #auxiliary classifier 2
        auxiliary2 = self._auxiliary(name = "auxiliary_2")(x)

        # inception(4e)
        x = self._inception_block(_1x1=0, _3x3r=128, _3x3=192, _d3x3r=192, _d3x3=256, _pool=0,
                                  name = "inception4e", strides = (2,2), pooling = "max")(x)

        #inception(5a)
        x = self._inception_block(_1x1=352, _3x3r=192, _3x3=320, _d3x3r=160, _d3x3=224, _pool=128, name = "inception5a")(x)
        #inception(5b)
        x = self._inception_block(_1x1=352, _3x3r=192, _3x3=320, _d3x3r=192, _d3x3=224, _pool=128, name = "inception5b")(x)

        x_shape = K.int_shape(x)
        x = AveragePooling2D(pool_size = (x_shape[self.row_axis], x_shape[self.col_axis]), strides=(1,1))(x)
        x = Flatten()(x)
        x = Dropout(0.4)(x)
        x = Dense(units = 1000, kernel_initializer = self.initializer)(x)
        x = BatchNormalization(axis = 1)(x)
        x = Activation("relu")(x)
        output_x = Dense(units = self.output_units, activation = "softmax", kernel_initializer=self.initializer, name = "main_output")(x)

        inceptionv2_model = Model(inputs = [input_x], outputs = [auxiliary1, auxiliary2, output_x])

        return inceptionv2_model


inception_builder = Inceptionv2_builder()
model = inception_builder.build_inception()
model.summary()