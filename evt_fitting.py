import libmr
import numpy as np
import scipy.spatial.distance as spd


def compute_mav_distances(activations, predictions, true_labels):
    """
    Calculates the mean activation vector (MAV) for each class and the distance to the mav for each vector.

    :param activations: logits for each image.
    :param predictions: predicted label for each image.
    :param true_labels: true label for each image.
    :return: MAV and euclidean-cosine distance to each vector.
    """
    correct_activations = list()
    mean_activations = list()
    eucos_dist = np.zeros(true_labels.shape[1])

    for cl in range(10):
        # Find correctly predicted samples and store activation vectors.
        i = (np.argmax(true_labels, 1) == predictions)
        i = i & (predictions == cl)
        act = activations[i, :]
        correct_activations.append(act)

        # Compute MAV for class.
        mean_act = np.mean(act, axis=0)
        mean_activations.append(mean_act)

        # Compute all, for this class, correctly classified images' distance to the MAV.
        for col in range(len(act)):
            eucos_dist[cl] = spd.euclidean(mean_act, act[col, :])/200. + spd.cosine(mean_act, act[col, :])

    return mean_activations, eucos_dist


def weibull_tailfitting(eucos_dist, mean_activations, taillength=8):
    """
    Fits a Weibull model of the logit vectors farthest from the MAV.

    :param eucos_dist: the euclidean-cosine distance from the MAV.
    :param mean_activations: mean activation vector (MAV).
    :param taillength:
    :return: weibull model.
    """

    weibull_model = {}
    for cl in range(10):
        weibull_model[str(cl)] = {}
        weibull_model[str(cl)]['eucos_distances'] = eucos_dist[cl]
        weibull_model[str(cl)]['mean_vec'] = mean_activations[cl]
        weibull_model[str(cl)]['weibull_model'] = []
        mr = libmr.MR(verbose=True)
        tailtofit = sorted(mean_activations[0])[-taillength:]
        mr.fit_high(tailtofit, len(tailtofit))
        weibull_model[str(cl)]['weibull_model'] = mr

    return weibull_model
