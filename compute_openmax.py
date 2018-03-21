from evt_fitting import *
import numpy as np


def compute_open_max_probability(openmax_known_score, openmax_unknown_score):
    """
    Compute the OpenMax probability.

    :param openmax_known_score: Weibull scores for known labels.
    :param openmax_unknown_score: Weibull scores for unknown unknowns.
    :return: OpenMax probability.
    """

    prob_scores, prob_unknowns, scores = [], [], []

    # Compute denominator for closet set + open set normalization.
    # Sum up the known classes scores.
    for category in range(10):
        scores += [np.exp(openmax_known_score[category])]
    total_denominator = np.sum(np.exp(openmax_known_score)) + np.exp(openmax_unknown_score)

    # Scores for image belonging to either closed set classes or open set.
    prob_scores = np.array([scores / total_denominator])
    prob_unknowns = np.array([np.exp(openmax_unknown_score) / total_denominator])

    openmax_scores = np.append(prob_scores.tolist(), prob_unknowns)

    assert len(openmax_scores) == 11
    return openmax_scores


def recalibrate_scores(weibull_model, img_layer_act, alpharank=10):
    """

    :param weibull_model: pre-computed Weibull model.
                          Dictionary with [class_labels]['euclidean distances', 'mean_vec', 'weibull_model']
    :param img_layer_act: activations in penultimate layer.
                          List with size [10,]
    :param alpharank: number of top classes to revise/check.
    :return: OpenMax probability of image.
    """

    num_labels = 10

    # Sort index of activations from highest to lowest. Size [10,]
    ranked_list = np.argsort(img_layer_act)
    ranked_list = np.ravel(ranked_list)
    ranked_list = ranked_list[::-1]  # reverse order

    # Obtain alpha weights for highest -> lowest activations.
    alpha_weights = [((alpharank + 1) - i) / float(alpharank) for i in range(1, alpharank + 1)]
    ranked_alpha = np.zeros(num_labels)
    for i in range(len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]     # Size [10,]

    # Calculate OpenMax probabilities
    openmax_penultimate, openmax_penultimate_unknown = [], []
    for categoryid in range(num_labels):  # 10 loops
        label_weibull = weibull_model[str(categoryid)]['weibull_model']  # Obtain the corresponding Weibull model.
        label_mav = weibull_model[str(categoryid)]['mean_vec']    # Obtain MAV for specific class. Size [10,]
        img_dist = np.linalg.norm(label_mav - img_layer_act)/200.  # Varför 200?

        weibull_score = label_weibull.w_score(img_dist)

        modified_layer_act = img_layer_act[0][categoryid] * (1 - weibull_score * ranked_alpha[categoryid])  # Revise av.
        openmax_penultimate += [modified_layer_act]  # Append revised av. to a total list. [10,1]
        openmax_penultimate_unknown += [(img_layer_act[0][categoryid] - modified_layer_act)]  # Av. 'unknown unknowns'.

    openmax_closedset_logit = np.asarray(openmax_penultimate)  # [10,1]
    openmax_openset_logit = np.asarray(np.sum(openmax_penultimate_unknown))  # [10,]

    # Pass the recalibrated penultimate layer scores for the image into OpenMax probability.
    openmax_probab = compute_open_max_probability(openmax_closedset_logit, openmax_openset_logit)

    return openmax_probab
