from model import *
from utilities import *
from compute_openmax import *
from evt_fitting import *

import tensorflow as tf
import numpy as np

# Load MNIST and, if omniglot_bool, Omniglot datasets.
x_train, y_train, x_val, y_val, x_test, y_test = load_datasets(test_size=100, val_size=5000, omniglot_bool=False,
                                                               name_data_set='data_mnist.h5', force=False,
                                                               create_file=True, r_seed=None)

# Build model.
tf.reset_default_graph()
sess = tf.Session()
net = MnistCNN(sess)

# Train model.
#net.train_model(x_train, y_train, x_val, y_val, epochs=5, verbose=1)

# Test model.
predictions, _, activations = net.predict(x_test)

# Evaluate accuracy.
accuracy = np.sum(np.argmax(y_test, 1) == predictions)
print(f'Test accuracy {accuracy/len(y_test)} %')

"""
# Print confusion matrices.
matrix1, matrix2 = load_confusion_matrix(predictions, y_test)
print("Full Confusion Matrix")
print(np.array_str(matrix1, precision=4, suppress_small=True))
print("Mnist-or-Glot Confusion Matrix")
print(np.array_str(matrix2, precision=4, suppress_small=True))
"""

# Pre-computations.
mean_activations, euclid_dist = compute_mav_distances(activations[-1], predictions, y_test)
weibull_model = weibull_tailfitting(euclid_dist, mean_activations)

# Test new image.
pred, _, act = net.predict([x_test[0]])

# Compute OpenMax probabilities on images.
openmax_probab = recalibrate_scores(weibull_model, act[-1], alpharank=10)

print("SoftMax prediction: {}".format(np.argmax(pred)))
print("OpenMax prediction: {}".format(np.argmax(openmax_probab)))
print("True label: {}".format(np.argmax(y_test[0])))




