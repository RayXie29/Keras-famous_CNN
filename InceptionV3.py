import keras
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, BatchNormalization, Activation
from keras.layers import Flatten, Input, concatenate, Dropout
from keras.regularizers import  l2
from keras import backend as K
from keras.models import Model


class Inceptionv3_builder():

    def __init__(self, input_shape = (299,299,3), output_units = 1000, init_strides = (2,2),
                 regularizer = l2(1e-4), initializer = "he_normal", init_maxpooling = True):

        '''
        :param input_shape: input shape of dataset
        :param output_units: output result dimension
        :param init_strides: The strides for first convolution layer
        :param regularizer: regularizer for all the convolution layers in whole NN
        :param initializer: weight/parameters initializer for all convolution & fc layers in whole NN
        :param init_maxpooling: Do the maxpooling after first two convolution layers or not
        '''

        assert len(input_shape) == 3, "input shape should be dim 3 ( row, col, channel or channel row col )"

        self.input_shape = input_shape
        self.output_units = output_units
        self.init_strides = init_strides
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

    def _cn_bn_relu(self, filters = 32, kernel_size = (3,3), strides = (1,1), padding = "same"):
        '''
        convenient function to build convolution -> batch_nromalization -> relu activation layers
        '''
        def f(input_x):

            x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                       kernel_regularizer=self.regularizer,kernel_initializer=self.initializer)(input_x)
            x = BatchNormalization(axis=self.channel_axis)(x)
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

            x = AveragePooling2D(pool_size=(5,5), strides = (3,3), padding = "same")(input_x)
            x = self._cn_bn_relu(filters = 128, kernel_size = (5,5), strides = (1,1), padding = "same")(x)
            x = Flatten()(x)
            x = Dense(units = 1024, kernel_initializer = self.initializer)(x)
            x = BatchNormalization(axis = 1)(x)
            x = Activation("relu")(x)
            x = Dropout(0.7)(x)

            return Dense(units = self.output_units , activation = "softmax", kernel_initializer=self.initializer, name = name)(x)

        return f


    def _inception_block35x35(self,_1x1 = 64, _3x3r = 48, _3x3 = 64, _d3x3r = 64, _d3x3 = 96, _pool = 64, name = "inception_fig5_1"):
        '''
        A function for building inception block of figure5 in original article,
        '''
        def f(input_x):

            branch1x1 = self._cn_bn_relu(filters = _1x1, kernel_size = (1,1))(input_x)

            branchpooling = AveragePooling2D(pool_size=(3,3), strides = (1,1), padding = "same")(input_x)
            branchpooling = self._cn_bn_relu(filters = _pool, kernel_size = (1,1))(branchpooling)

            branch3x3 = self._cn_bn_relu(filters = _3x3r, kernel_size = (1,1))(input_x)
            branch3x3 = self._cn_bn_relu(filters = _3x3, kernel_size = (3,3))(branch3x3)

            dbranch3x3 = self._cn_bn_relu(filters = _d3x3r, kernel_size = (1,1))(input_x)
            dbranch3x3 = self._cn_bn_relu(filters = _d3x3, kernel_size = (3,3))(dbranch3x3)
            dbranch3x3 = self._cn_bn_relu(filters = _d3x3, kernel_size = (3,3))(dbranch3x3)

            return concatenate([branch1x1, branchpooling, branch3x3, dbranch3x3], axis = self.channel_axis, name = name)

        return f

    def _GridSizeReduction35x35(self, _3x3r = 288, _3x3 = 384, _d3x3r = 64, _d3x3 = 96):
        '''
        A function for dimension reducing from 35x35 -> 17x17
        '''
        def f(input_x):

            branchpool = AveragePooling2D(pool_size=(3,3), strides = (2,2), padding = "valid")(input_x)

            branch3x3 = self._cn_bn_relu(filters = _3x3r, kernel_size = (1,1))(input_x)
            branch3x3 = self._cn_bn_relu(filters = _3x3,  kernel_size = (3,3), strides = (2,2), padding = "valid")(branch3x3)

            dbranch3x3 = self._cn_bn_relu(filters = _d3x3r, kernel_size = (1,1))(input_x)
            dbranch3x3 = self._cn_bn_relu(filters = _d3x3,  kernel_size = (3,3))(dbranch3x3)
            dbranch3x3 = self._cn_bn_relu(filters = _d3x3,  kernel_size = (3,3), strides = (2,2), padding = "valid")(dbranch3x3)

            return concatenate([branchpool, branch3x3, dbranch3x3], axis = self.channel_axis)

        return f

    def _inception_block17x17(self, _1x1 = 192, _7x7r = 128, _7x7 = 192, _d7x7r = 128, _d7x7 = 192, _pool = 192, name = "inception_fig6_1"):
        '''
        A function for building inception block of figure6 in original article,
        '''
        def f(input_x):

            branch1x1 = self._cn_bn_relu(filters=_1x1, kernel_size=(1, 1))(input_x)

            branchpooling = AveragePooling2D(pool_size = (3,3), strides = (1,1), padding = "same")(input_x)
            branchpooling = self._cn_bn_relu(filters = _pool, kernel_size = (1,1))(branchpooling)

            branch7x7 = self._cn_bn_relu(filters = _7x7r, kernel_size = (1,1))(input_x)
            branch7x7 = self._cn_bn_relu(filters = _7x7r, kernel_size = (7,1))(branch7x7)
            branch7x7 = self._cn_bn_relu(filters = _7x7, kernel_size=(1, 7))(branch7x7)

            dbranch7x7 = self._cn_bn_relu(filters = _d7x7r, kernel_size = (1,1))(input_x)
            for i in range(2):
                dbranch7x7 = self._cn_bn_relu(filters = _d7x7r, kernel_size=(7, 1))(branch7x7)

                if i == 0:
                    dbranch7x7 = self._cn_bn_relu(filters=_d7x7r, kernel_size=(1, 7))(branch7x7)
                else :
                    dbranch7x7 = self._cn_bn_relu(filters=_d7x7, kernel_size=(1, 7))(branch7x7)


            return concatenate([branch1x1, branchpooling, branch7x7, dbranch7x7], axis = self.channel_axis, name = name)

        return f


    def _GridSizeReduction17x17(self, _3x3r = 192, _3x3 =320, _d7x7x3r = 192, _d7x7x3 = 192):
        '''
        A function for dimension reducing from 17x17 -> 8x8
        '''
        def f(input_x):

            branchpool = AveragePooling2D(pool_size = (3,3), strides = (2,2), padding = "valid")(input_x)

            branch7x7 = self._cn_bn_relu(filters = _3x3r, kernel_size = (1,1))(input_x)
            branch7x7 = self._cn_bn_relu(filters = _3x3 , kernel_size = (3,3), strides = (2,2), padding = "valid")(branch7x7)

            dbranch7x7 = self._cn_bn_relu(filters = _d7x7x3r, kernel_size = (1, 1))(input_x)
            dbranch7x7 = self._cn_bn_relu(filters = _d7x7x3, kernel_size = (7, 1))(dbranch7x7)
            dbranch7x7 = self._cn_bn_relu(filters = _d7x7x3, kernel_size = (1, 7))(dbranch7x7)
            dbranch7x7 = self._cn_bn_relu(filters = _d7x7x3,  kernel_size = (3, 3), strides = (2, 2), padding = "valid")(dbranch7x7)

            return concatenate([branchpool, branch7x7, dbranch7x7], axis = self.channel_axis)

        return f

    def _inception_block8x8(self, _1x1 = 320, _pool = 192, _3x3r = 384, _3x3 = 384, _d3x3r = 448, _d3x3 = 384, name = "inception_fig7_1"):
        '''
        A function for building inception block of figure7 in original article,
        '''

        def f(input_x):

            branch1x1 = self._cn_bn_relu(filters = _1x1, kernel_size = (1,1))(input_x)

            branchpool = AveragePooling2D(pool_size = (3,3), strides = (1,1), padding = "same")(input_x)
            branchpool = self._cn_bn_relu(filters = _pool, kernel_size = (1,1))(branchpool)

            branch3x3 = self._cn_bn_relu(filters = _3x3r, kernel_size = (1,1))(input_x)
            branch3x3_1 = self._cn_bn_relu(filters = _3x3, kernel_size = (3,1))(branch3x3)
            branch3x3_2 = self._cn_bn_relu(filters = _3x3, kernel_size = (1,3))(branch3x3)

            dbranch3x3 = self._cn_bn_relu(filters = _d3x3r, kernel_size = (1,1))(input_x)
            dbranch3x3 = self._cn_bn_relu(filters = _d3x3, kernel_size = (3,3))(dbranch3x3)
            dbranch3x3_1 = self._cn_bn_relu(filters = _d3x3, kernel_size = (3,1))(dbranch3x3)
            dbranch3x3_2 = self._cn_bn_relu(filters = _d3x3, kernel_size = (1,3))(dbranch3x3)

            return concatenate([branch1x1, branchpool, branch3x3_1, branch3x3_2, dbranch3x3_1, dbranch3x3_2], axis = self.channel_axis, name = name)

        return f


    def build_inception(self):

        '''
        Main function for building inceptionV3 nn
        :return: An inceptionV3 nn
        '''

        input_x = Input(self.input_shape)


        #Few traditional convolutional layers at lower layers
        #Which are factorized by original 7x7 convolution layer
        x = self._cn_bn_relu(filters = 32, kernel_size = (3,3), strides = self.init_strides, padding = "valid")(input_x)
        x = self._cn_bn_relu(filters = 32, kernel_size = (3,3), strides = (1,1), padding = "valid")(x)
        x = self._cn_bn_relu(filters = 64, kernel_size = (3,3), strides=(1,1), padding="same")(x)

        if self.init_maxpooling:
            x = MaxPooling2D(pool_size = (3,3), strides = (2,2), padding = "valid")(x)

        x = self._cn_bn_relu(filters = 80, kernel_size = (3,3), strides=(1,1), padding = "valid")(x)
        x = self._cn_bn_relu(filters = 192, kernel_size = (3,3), strides = self.init_strides, padding = "valid")(x)
        x = self._cn_bn_relu(filters = 288, kernel_size = (3,3), strides = (1,1), padding = "same")(x)

        #First 3 inception block, which are using architecture of figure5 in original article
        for i in range(3):
            x = self._inception_block35x35(_1x1=64,_3x3r=48,_3x3=64,_d3x3r=64,_d3x3=96, name = "inception_fig5_"+str(i+1))(x)

        #Dimension reducing #1 (from 35x35 -> 17x17 in original article)
        x = self._GridSizeReduction35x35( _3x3r = 288, _3x3 = 384, _d3x3r = 64, _d3x3 = 96)(x)

        # 5 inception block, which are using architecture of figure6 in original article
        for i in range(5):
            x = self._inception_block17x17(_1x1=192,_7x7r=128,_7x7=192,_d7x7r=128,_d7x7=192,_pool=192, name = "inception_fig6_"+str(i+1))(x)

        # auxiliary classifier
        auxiliary = self._auxiliary(name = "auxiliary_1")(x)

        #Dimension reducing #2 (from 17x17 -> 8x8 in original article)
        x = self._GridSizeReduction17x17(_3x3r=192,_3x3=320,_d7x7x3r=192,_d7x7x3=192)(x)

        for i in range(2):
            x = self._inception_block8x8(_1x1 = 320, _pool = 192, _3x3r = 384, _3x3 = 384, _d3x3r = 448, _d3x3 = 384, name = "inception_fig7_"+str(i+1))(x)

        x_shape = K.int_shape(x)

        x = AveragePooling2D(pool_size = (x_shape[self.row_axis], x_shape[self.col_axis]), strides = (1,1))(x)
        x = Flatten()(x)
        x = Dense(units = 2048, kernel_initializer=self.initializer)(x)
        x = BatchNormalization(axis = 1)(x)
        x = Activation("relu")(x)

        output_x = Dense(units = self.output_units, activation = "softmax", kernel_initializer=self.initializer,name = "main_output")(x)

        inceptionv3_model = Model(inputs = [input_x], outputs = [output_x,auxiliary])
        return inceptionv3_model

inception_builder = Inceptionv3_builder()
model = inception_builder.build_inception()
model.summary()