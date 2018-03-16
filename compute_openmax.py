from evt_fitting import *
import numpy as np


def compute_open_max_probability(openmax_fc8, openmax_score_u):
    """

    :param openmax_fc8:
    :param openmax_score_u:
    :return:
    """

    prob_scores, prob_unknowns, scores = [], [], []

    for category in range(10):
        scores += [np.exp(openmax_fc8[category])]
    total_denominator = np.sum(np.exp(openmax_fc8)) + np.exp(openmax_score_u)
    prob_scores = np.array([scores / total_denominator])
    prob_unknowns = np.array([np.exp(openmax_score_u) / total_denominator])

    mean_scores = np.mean(prob_scores, axis=0)
    mean_unknowns = np.mean(prob_unknowns, axis=0)
    modified_scores = mean_scores.tolist()
    modified_scores = np.add(modified_scores, mean_unknowns)

    assert len(modified_scores) == 10
    return modified_scores


def recalibrate_scores(weibull_model, img_layer_activations, alpharank=10):
    """

    :param weibull_model: pre-computed weibull model.
                          Dictionary with [class_labels]['euclidean distances', 'mean_vec', 'weibull_model']
    :param activations: activations of image in the penultimate layer.
                        List with size [10,]
    :param alpharank: number of top classes to revise/check.
    :return:
    """

    num_labels = 10

    img_layer_act = img_layer_activations  # Activations in penultimate layer

    # Sort index of activations from highest to lowest. Size [10,]
    ranked_list = np.argsort(img_layer_act)
    ranked_list = np.ravel(ranked_list)
    ranked_list = ranked_list[::-1]  # reverse order

    alpha_weights = [((alpharank + 1) - i) / float(alpharank) for i in range(1, alpharank + 1)]
    # Obtain alpha weights for highest -> lowest activations.
    ranked_alpha = np.zeros(num_labels)
    for i in range(len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]     # Size [10,]

    # Calculate OpenMax probabilities
    openmax_penultimate, openmax_penultimate_unknown = [], []
    for categoryid in range(num_labels):  # 10 loops
        label_weibull = weibull_model[str(categoryid)]['weibull_model']  # Obtain the corresponding weibull model.
        label_mav = weibull_model[str(categoryid)]['mean_vec']    # Obtain MAV for specific class. Size [10,]
        img_dist = np.linalg.norm(label_mav - img_layer_act)/200.  # Calculate input image's distance from label MAV. Scalar. Varför 200?

        weibull_score = label_weibull.w_score(img_dist)  # Obtain weibull score for image. Scalar.

        modified_layer_act = img_layer_act[0][categoryid] * (1 - weibull_score * ranked_alpha[categoryid])  # Revise activation vector. Scalar.
        openmax_penultimate += [modified_layer_act]  # Append revised a.v. to a total list. [10,1]
        openmax_penultimate_unknown += [(img_layer_act[0][categoryid] - modified_layer_act)]  # Define a.v. for 'unknown unknowns'. Scalar.

    openmax_closedset_logit = np.asarray(openmax_penultimate)  # [10,1]
    openmax_openset_logit = np.asarray(np.sum(openmax_penultimate_unknown))  # [10,]

    # Pass the recalibrated fc8 scores for the image into openmax
    openmax_probab = compute_open_max_probability(openmax_closedset_logit, openmax_openset_logit)  # Probability.

    return openmax_probab
