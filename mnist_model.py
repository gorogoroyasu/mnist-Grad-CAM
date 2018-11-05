import keras
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from keras.layers.core import Activation

class Model:
    def __init__(self):
        self.build_model()

    def build_model(self):
        self.inputs = Input(shape=(28, 28, 1, ),name='imgs')
        self.labels = Input((10,),name='labels')
        self.aa = Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         )(self.inputs)
        self.bb = MaxPooling2D(pool_size=(2, 2))(self.aa)
        self.cc = Conv2D(64, (3, 3), activation='relu')(self.bb)
        self.dd = MaxPooling2D(pool_size=(2, 2))(self.cc)
        self.ee = Conv2D(128, (3, 3), activation='relu')(self.dd)
        self.a = Conv2D(128, (3, 3), activation='relu')(self.dd)
        self.b = MaxPooling2D(pool_size=(2, 2))(self.a)

        self.c = Flatten()(self.a)

        self.d = Dense(64, activation='relu')(self.c)
        self.e = Dropout(0.5)(self.d)
        self.before_soft_max = Dense(10)(self.e)
        self.predictions = Activation('softmax')(self.before_soft_max)

        self.g = Lambda(lambda x: K.gradients(x[0] * x[2], x[1]), output_shape=list(self.a.shape))([self.before_soft_max, self.a, self.labels])
        self.cost = Lambda(lambda x: (-1) * K.sum(x[0] * K.log(x[1]), axis=1), output_shape=list(self.labels.shape))([self.labels, self.predictions])
        self.gb_grad = Lambda(lambda x: K.gradients(x[0], x[1]), output_shape=list(self.inputs.shape))([self.cost, self.inputs])
