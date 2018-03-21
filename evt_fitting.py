import libmr
import numpy as np
import scipy.spatial.distance as spd


def compute_mav_distances(activations, predictions, true_labels):
    """

    :param activations:
    :param predictions:
    :param true_labels:
    :return:
    """
    correct_activations = list()
    mean_activations = list()
    euclid_dist = np.zeros(true_labels.shape[1])

    for cl in range(10):
        # Find correctly predicted activations.
        i = (np.argmax(true_labels, 1) == predictions)
        i = i & (predictions == cl)
        act = activations[i, :]
        correct_activations.append(act)

        # Compute mean activations vector, MAV, for class.
        mean_act = np.mean(act, axis=0)
        mean_activations.append(mean_act)

        # Compute all correctly classified images, of this class, distances to the MAV.
        for col in range(len(act)):
            euclid_dist[cl] = spd.euclidean(mean_act, act[col, :])/200. + spd.cosine(mean_act, act[col, :])

    return mean_activations, euclid_dist


def weibull_tailfitting(euclid_dist, mean_activations):
    """

    :param euclid_dist:
    :param mean_activations:
    :return:
    """
    weibull_model = {}
    for cl in range(10):
        # Fit the 20 activations farthest from the MAV.
        weibull_model[str(cl)] = {}
        weibull_model[str(cl)]['euclidean_distances'] = euclid_dist[cl]
        weibull_model[str(cl)]['mean_vec'] = mean_activations[cl]
        weibull_model[str(cl)]['weibull_model'] = []
        mr = libmr.MR()
        tailtofit = sorted(mean_activations[0])[-5:]
        mr.fit_high(tailtofit, len(tailtofit))
        weibull_model[str(cl)]['weibull_model'] = mr

    return weibull_model
