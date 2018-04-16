# Implementation of OpenMax on Mnist and Omniglot
This is work for the Master's thesis project "Input Verification of Deep Neural Networks". The purpose is to use OpenMax to successfully reject Omniglot images from a CNN trained on Mnist. The algorithm parameters is tuned on adversarial images.

### Requirements
Cython, LibMR, Numpy, Scipy, h5py, matplotlib

## Usage
### Pre-computations
To compute the mean activation vector, mav, for each class and the euclidean distances between each sample activation vector and the mav, run
```
mean_activations, eucos_dist = compute_mav_distances(activations[-1],predictions, y_train)
```
where activations[-1] are the activation vector from the penultimate layer for each Mnist image. The predictions are the predicted labels and y_train the true labels.

The 'mav_eucos.h5' file holds mean activation vectors, MAV, for the penultimate layer activations and the distance from each activation vector to the MAV when fed 50000 Mnist images.

### Compute Weibull models
Use the mavs and distances to compute weibull models for each class. This is done by using LibMR meta-recognition which, given a tail length, fits the activation vectors farthest from their corresponding mav. The tail length decides how many vectors to be used. Run
```
weibull_models = weibull_tailfitting(eucos_dist, mean_activations, tail_length)
```
to receive ten weibull models; one for each class.

### Evaluate
Given a test image, the image is run through the CNN and the resulting penultimate activations, act[-1], are saved. The OpenMax probailities are then computed by calling
```
openmax_probab = recalibrate_scores(weibull_models, act[-1], alpharank=10)
```
where weibull_models are the weibull models for each class. The alpharank sets how many top-level classes that should be considered. As Mnist only has ten classes, it is here set to include all of them. If the OpenMax probability is highest for class eleven (open-set class) or if the highest probability is lower than a certain threshold value (chosen from adversarial images tuning set) the image should be rejected.

## Acknowledgments
This is an implementation the OpenMax layer algorithm from A. Bendale, T. Boult “Towards Open Set Deep Networks” IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

**References:** http://vast.uccs.edu/~abendale/papers/0348.pdf

The implementation is based upon https://github.com/abhijitbendale/OSDN with the difference that is the use of a CNN trained on Mnist images, tuned on adversarial images and tested on Omniglot images.