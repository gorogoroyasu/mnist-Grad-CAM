FROM tensorflow/tensorflow:1.12.0-py3

# enable cv2
RUN apt-get update -y && apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install opencv-python scikit-image
