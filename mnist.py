import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model

batch_size = 1000
num_classes = 10
epochs = 60
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

img_rows, img_cols = 28, 28
(x_train, y_train_original), (x_test, y_test_original) = mnist.load_data()
import numpy as np
y_train_original = y_train_original.astype(np.int16)
y_test_original = y_test_original.astype(np.int16)

x_train = x_train / 255.
x_test = x_test / 255.

from mnist_model import Model as MM
y_train = keras.utils.to_categorical(y_train_original, num_classes)
y_test = keras.utils.to_categorical(y_test_original, num_classes)

x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

m = MM()

model=Model(inputs=[m.labels, m.inputs], outputs=[m.predictions])
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit([y_train, x_train], y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=([y_test, x_test], y_test))

model.save('my_model.h5')
model.summary()
