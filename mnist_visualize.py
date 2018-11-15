import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import sys, cv2
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from mnist_model import Model as MM
from pathlib import Path

# create mnist_cams directory. 
# @todo: make the directory name changable with args.
path = Path('mnist_cams')
path.mkdir(exist_ok=True)
del path

K.set_learning_phase(1)
img_rows, img_cols = 300, 400
num_classes = 10

m = MM()

from tensorflow.keras.models import Model
model=Model(inputs=[m.labels, m.inputs], outputs=[m.predictions, m.g, m.a, m.gb_grad])
model.summary()
model.load_weights('my_model.h5')

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.
x_test = x_test.reshape((-1, 28, 28, 1))
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_test.shape)
for target_y_train_num in range(20):

    result = model.predict([
        y_test[target_y_train_num].reshape((1, 10)),
        x_test[target_y_train_num].reshape((1, 28, 28, 1))
    ])
    print('answer: ', K.eval(K.argmax(y_test[target_y_train_num])))
    print('prediction: ', K.eval(K.argmax(result[0])))

    conv_grad = result[1]
    conv_grad = conv_grad.reshape(conv_grad.shape[1:])
    conv_output = result[2]
    conv_output = conv_output.reshape(conv_output.shape[1:])
    input_grad = result[3]
    input_grad = input_grad.reshape(input_grad.shape[1:])
    gradRGB = gb_viz = input_grad

    from skimage.transform import resize
    import cv2

    # global average pooling
    weights = np.mean(conv_grad, axis = (0, 1))
    cam = np.zeros(conv_output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * conv_output[:, :, i]

    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    cam = resize(cam, (28,28), preserve_range=True)

    img = x_test[target_y_train_num].astype(float)
    img -= np.min(img)
    img /= img.max()

    cam_heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)


    cam = np.float32(cam.reshape((28, 28, 1))) * np.float32(img)
    cam = 255 * cam / np.max(cam)
    cam = np.uint8(cam)

    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    plt.figure()
    img_int = (img * 255.).astype(int).reshape(img.shape[:2])
    plt.gray()
    plt.imshow(img_int)
    plt.savefig('mnist_cams/original_{}.png'.format(target_y_train_num))
    plt.close()

    plt.figure()
    plt.imshow(cam_heatmap)
    plt.savefig('mnist_cams/heatmap_{}.png'.format(target_y_train_num))
    plt.close()

    plt.figure()
    plt.imshow(img_int)
    plt.imshow(cam_heatmap, alpha=0.5)
    plt.savefig('mnist_cams/heatmap_overlaied_{}.png'.format(target_y_train_num))
    plt.close()

    gb_viz -= np.min(gb_viz)
    gb_viz /= gb_viz.max()
    img_int = (gb_viz * 255.).astype(int).reshape(img.shape[:2])
    imgplot = plt.imshow(img_int)
    plt.savefig('mnist_cams/grad-cam-backpropagation_{}.png'.format(target_y_train_num))
    plt.close()

    gd_gb = gb_viz * cam
    img_int = (gd_gb * 255.).astype(int).reshape(img.shape[:2])
    imgplot = plt.imshow(img_int)
    plt.savefig('mnist_cams/guided-grad-cam_{}.png'.format(target_y_train_num))
    plt.close()
