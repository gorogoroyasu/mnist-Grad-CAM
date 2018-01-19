mnist-Grad-CAM

# contents

I implemented [Grad-CAM](https://arxiv.org/abs/1610.02391) and applied it to mnist datasets.
if you find some bugs or anything else, feel free to open issues or PRs.

# environment

```
Python 3.6.2
Keras==2.0.9
numpy==1.13.3
scikit-image==0.13.1
h5py==2.7.1
etc
```

# usage

all you need is the commands shown below.
```
mkdir project
git clone git@github.com:gorogoroyasu/mnist-Grad-CAM.git project && cd project
python mnist.py
python mnist_visualize.py
```
#examples

## original
![image](https://github.com/gorogoroyasu/mnist-Grad-CAM/blob/master/examples/original.png?raw=true)

## heatmap
![image](https://github.com/gorogoroyasu/mnist-Grad-CAM/blob/master/examples/heatmap.png?raw=true)

## heatmap overrayed
![image](https://github.com/gorogoroyasu/mnist-Grad-CAM/blob/master/examples/heatmap_overlaied.png?raw=true)

## guided grad cam figure
![image](https://github.com/gorogoroyasu/mnist-Grad-CAM/blob/master/examples/guided-grad-cam.png?raw=true)
# reference

### CNN structure
https://github.com/keras-team/keras/blob/950e5d063320a72ca61f2082c154a65a48766239/examples/mnist_cnn.py

### Grad-CAM implements
https://github.com/insikk/Grad-CAM-tensorflow  
https://github.com/jacobgil/keras-grad-cam  
https://github.com/ysasaki6023/imageCounting  
