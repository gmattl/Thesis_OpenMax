from evt_fitting import *
import numpy as np
#weibull_model = evt_fitting(activations=, predictions=, true_labels=)


def compute_open_max_probability(openmax_fc8, openmax_score_u):
    """

    :param openmax_fc8:
    :param openmax_score_u:
    :return:
    """

    prob_scores, prob_unknowns, scores = [], [], []

    for category in range(10):
        scores += [np.exp(openmax_fc8[category])]
    total_denominator = np.sum(np.exp(openmax_fc8)) + np.exp(np.sum(openmax_score_u))
    prob_scores = np.array([scores / total_denominator])
    prob_unknowns = np.array([np.exp(np.sum(openmax_score_u)) / total_denominator])

    mean_scores = np.mean(prob_scores, axis=0)
    mean_unknowns = np.mean(prob_unknowns, axis=0)
    modified_scores = mean_scores.tolist()
    modified_scores = np.add(modified_scores, mean_unknowns)

    assert len(modified_scores) == 11
    return modified_scores


def recalibrate_scores(weibull_model, labels, predictions, activations, alpharank = 10):
    """

    :param weibull_model: pre-computed weibull model
    :param labels: all labels
    :param predictions: prediction of image
    :param activations:
    :param alpharank:
    :return:
    """

    img_layer_act = activations[-1, :]  # activations in penultimate layer
    ranked_list = activations.np.argsort().np.ravel()[::-1]     # sort index of activations from highest -> lowest
    alpha_weights = [((alpharank + 1) - i) / float(alpharank) for i in range(1, alpharank + 1)]

    # Obtain alpha weights for highest -> lowest activations.
    ranked_alpha = np.zeros(len(labels))
    for i in range(len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]

    # Calculate OpenMax probabilities
    openmax_penultimate, openmax_penultimate_unknown = [], []
    for categoryid in range(len(labels)):
        label_weibull = weibull_model[str(categoryid)]  # Obtain the corresponding weibull model for the class label.
        img_dist = compute_distances(activations, predictions, labels)  # Calculate input image's distance from MAV

        weibull_score = label_weibull.w_score(img_dist)  # Obtain weibull score for image.
        modified_layer_score = img_layer_act * (1 - weibull_score * ranked_alpha[categoryid])
        openmax_penultimate += [modified_layer_score]
        openmax_penultimate_unknown += [img_layer_act - modified_layer_score]
    openmax_fc8 = np.asarray(openmax_penultimate)
    openmax_score_u = np.asarray(openmax_penultimate_unknown)

    # Pass the recalibrated fc8 scores for the image into openmax
    openmax_probab = compute_open_max_probability(openmax_fc8, openmax_score_u)
    softmax_probab = np.argmax(predictions, 1)

    return openmax_probab, softmax_probab



