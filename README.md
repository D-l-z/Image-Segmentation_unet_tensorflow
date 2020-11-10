# Implementation of deep learning framework -- Unet, using Keras

The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

---

### Data
### download data
The original dataset is from [isbi challenge](http://brainiac2.mit.edu/isbi_challenge/)
Download the dataset and save them to a fold named data/membrane

### Data augmentation

The data for training contains 30 512*512 images, which are far not enough to feed a deep learning neural network. I use a module called ImageDataGenerator in keras.preprocessing.image to do data augmentation.


### Training

The model is trained for 50 epochs.

After 5 epochs, calculated accuracy is about 0.92.

Loss function for the training is basically just a binary crossentropy.

---

### Dependencies

This tutorial depends on the following libraries:

* Tensorflow
* Keras >= 1.0

Also, this code should be compatible with Python versions 2.7-3.5.


### Run main.py

You will see the predicted results of test image in data/membrane/test





