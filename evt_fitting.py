import libmr
import numpy as np


def compute_distances(activations, predictions, true_labels):
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
        act = activations[-1][i, :]
        correct_activations.append(act)

        # Compute mean activations vector, MAV, for class.
        mean_act = np.mean(act, axis=0)
        mean_activations.append(mean_act)

        # Compute distances from MAV.
        euclid_dist[cl] = np.linalg.norm(mean_act-act)


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
        tailtofit = sorted(mean_activations[0])[-20:]
        mr.fit_high(tailtofit, len(tailtofit))
        weibull_model[str(cl)]['weibull_model'] += [mr]

        print("Weibull fitting on activations done.")
    return weibull_model
