mnist-Grad-CAM

# contents

I implemented [Grad-CAM](https://arxiv.org/abs/1610.02391) and applied it to mnist datasets.
if you find some bugs or anything else, feel free to open issues or PRs.

# environment
`Dockerfile` and `docker-compose.yml` are settled in `docker/cpu` and `docker/gpu`.
They will make the executable environment on your machine by using docker.
when you use GPU, you can choose `docker/gpu` dir, otherwise you have to use `docker/cpu`.
The `docker/gpu` contents were well worked in my env, 
```
Ubuntu:16.04 
Geforce 1080Ti
```
If you find some issues, please tell it to me via issues.

`docker/cpu` worked fine on my local machine, which is Mac Book Pro 2016 Mid 2015.

# usage
```
cd docker/gpu(or docker/cpu)
docker-compose up -d --build
docker-compose exec tf bash
# train
python mnist.py
# visualization: my_model.h5 created by the command above.
python mnist_visualize.py
```
the visualized image will be putted in `mnist_cams` dir, which will be automatically created.

currently, this directory and `my_model.h5` names cannot be fixed by argments.  
it is my future work.

#examples

## original
![image](https://github.com/gorogoroyasu/mnist-Grad-CAM/blob/master/examples/original.png?raw=true)

## heatmap
![image](https://github.com/gorogoroyasu/mnist-Grad-CAM/blob/master/examples/heatmap.png?raw=true)

## heatmap overrayed
![image](https://github.com/gorogoroyasu/mnist-Grad-CAM/blob/master/examples/heatmap_overlayed.png?raw=true)

## guided grad cam figure
![image](https://github.com/gorogoroyasu/mnist-Grad-CAM/blob/master/examples/guided-grad-cam.png?raw=true)
# reference

### CNN structure
https://github.com/keras-team/keras/blob/950e5d063320a72ca61f2082c154a65a48766239/examples/mnist_cnn.py

### Grad-CAM implements
https://github.com/insikk/Grad-CAM-tensorflow  
https://github.com/jacobgil/keras-grad-cam  
https://github.com/ysasaki6023/imageCounting  
