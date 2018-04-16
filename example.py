import sys
sys.path.append('../..')
from cnn import MnistCNN
sys.path.append('../../Thesis_Utilities/')
from utilities import load_datasets, load_confusion_matrix
sys.path.append('../Thesis_OpenMax')
from compute_openmax import *
from evt_fitting import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import h5py

# Load datasets.
x_train, y_train, _, _, x_test, y_test = load_datasets(test_size=10000, val_size=30000, omniglot_bool=True,
                                                               name_data_set='data_omni.h5', force=False,
                                                               create_file=True, r_seed=1337)

# PARAMETERS.
# TODO: These are only allowed to be chosen for tuning set.
tail = 8
threshold = 0.43

# Limit the data to facilitate runs on slower computers.
x_test = x_test[9500:10500]
y_test = y_test[9500:10500]

# Build model.
tf.reset_default_graph()
sess = tf.Session()
model = MnistCNN(sess, save_dir='../../MnistCNN_save/')

# Test model on training data set for OpenMax build/ Weibull fitting.
#predictions, _, activations = model.predict(x_train)

# Pre-computations.
#mean_activations, eucos_dist = compute_mav_distances(activations[-1], predictions, y_train)
"""
f = h5py.File('../mav_eucos.h5', 'w')
f.create_dataset("mavs", data=mean_activations)
f.create_dataset("eucos", data=eucos_dist)
f.close()
"""
f = h5py.File('../mav_eucos.h5', 'r')
mean_activations = f['mavs'][:]
eucos_dist = f['eucos'][:]
f.close()

# Compute Weibull models.
print('Weibull tail fitting tail length: {}'.format(tail))
weibull_models = weibull_tailfitting(eucos_dist, mean_activations, taillength=tail)

# Testing.
acc = 0
acc2 = 0
open_probs = []
pred_ = np.zeros(len(y_test))
for i in range(0, len(x_test)):
    test_image = x_test[i:i + 1]
    test_label = y_test[i:i + 1]
    pred, _, act = model.predict(test_image)

    # Compute OpenMax probabilities on images.
    openmax_probab = recalibrate_scores(weibull_models, act[-1], alpharank=10)
    open_probs.append(list(openmax_probab))

    # Compute accuracy and Omniglot-or-not accuracy
    if np.argmax(openmax_probab) == np.argmax(test_label):
        acc2 = acc2 + 1
    if np.max(openmax_probab) < threshold:
        openmax_probab[10] = 1
    pred_[i] = (np.argmax(openmax_probab))
    if pred_[i] == 10:
        open_ = 1
    else:
        open_ = 0
    if np.squeeze(np.nonzero(test_label[0])) == 10:
        true = 1
    else:
        true = 0
    if open_ == true:
        acc = acc + 1

print("OpenMax per-label accuracy: {}".format(acc2 / len(x_test)))
print('OpenMax probability rejection threshold: {}'.format(threshold))
print("OpenMax novelty-or-not accuracy: {}".format(acc / len(x_test)))

conf_matrix = load_confusion_matrix(pred_, y_test)
print(conf_matrix)

xlin = np.linspace(0, len(x_test), np.int(len(x_test)))
plt.plot(xlin[:np.int(len(xlin)/2)], np.max(open_probs[:np.int(len(x_test)/2)], 1), '*', label='Closed set')
plt.plot(xlin[np.int(len(xlin)/2):], np.max(open_probs[np.int(len(x_test)/2):], 1), 'o', label='Open set')
plt.plot([0, np.max(xlin)], [threshold, threshold], 'k', label='Threshold')
plt.legend()
plt.title('Max Probabilities of OpenMax')
plt.xlabel('Samples')
plt.ylabel('Probability')
plt.show()




