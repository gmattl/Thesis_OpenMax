from evt_fitting import *
import numpy as np
import scipy.spatial.distance as spd


def compute_open_max_probability(openmax_known_score, openmax_unknown_score):
    """
    Compute the OpenMax probability.

    :param openmax_known_score: Weibull scores for known labels.
    :param openmax_unknown_score: Weibull scores for unknown unknowns.
    :return: OpenMax probability.
    """

    prob_closed, prob_open, scores = [], [], []

    # Compute denominator for closet set + open set normalization.
    # Sum up the class scores.
    for category in range(10):
        scores += [np.exp(openmax_known_score[category])]
    total_denominator = np.sum(np.exp(openmax_known_score)) + np.exp(openmax_unknown_score)

    # Scores for image belonging to either closed or open set.
    prob_closed = np.array([scores / total_denominator])
    prob_open = np.array([np.exp(openmax_unknown_score) / total_denominator])

    probs = np.append(prob_closed.tolist(), prob_open)

    assert len(probs) == 11
    return probs


def recalibrate_scores(weibull_model, img_layer_act, alpharank=10):
    """
    Computes the OpenMax probabilities of an input image.

    :param weibull_model: pre-computed Weibull model.
                          Dictionary with [class_labels]['euclidean distances', 'mean_vec', 'weibull_model']
    :param img_layer_act: activations in penultimate layer.
    :param alpharank: number of top classes to revise/check.
    :return: OpenMax probabilities of image.
    """

    num_labels = 10

    # Sort index of activations from highest to lowest.
    ranked_list = np.argsort(img_layer_act)
    ranked_list = np.ravel(ranked_list)
    ranked_list = ranked_list[::-1]

    # Obtain alpha weights for highest -> lowest activations.
    alpha_weights = [((alpharank + 1) - i) / float(alpharank) for i in range(1, alpharank + 1)]
    ranked_alpha = np.zeros(num_labels)
    for i in range(0, len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]

    # Calculate OpenMax probabilities
    openmax_penultimate, openmax_penultimate_unknown = [], []
    for categoryid in range(num_labels):
        label_weibull = weibull_model[str(categoryid)]['weibull_model']  # Obtain the corresponding Weibull model.
        label_mav = weibull_model[str(categoryid)]['mean_vec']    # Obtain MAV for specific class.
        img_dist = spd.euclidean(label_mav, img_layer_act)/200. + spd.cosine(label_mav, img_layer_act)

        weibull_score = label_weibull.w_score(img_dist)

        modified_layer_act = img_layer_act[0][categoryid] * (1 - weibull_score * ranked_alpha[categoryid])  # Revise av.
        openmax_penultimate += [modified_layer_act]  # Append revised av. to a total list.
        openmax_penultimate_unknown += [img_layer_act[0][categoryid] - modified_layer_act]  # A.v. 'unknown unknowns'.

    openmax_closedset_logit = np.asarray(openmax_penultimate)
    openmax_openset_logit = np.sum(openmax_penultimate_unknown)

    # Transform the recalibrated penultimate layer scores for the image into OpenMax probability.
    openmax_probab = compute_open_max_probability(openmax_closedset_logit, openmax_openset_logit)

    return openmax_probab
