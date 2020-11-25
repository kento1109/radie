import numpy as np

# def least_confident(x):
#     """
#     :param x: probability matrix (n_words, n_labels)
#     :return: uncertainty measure (scalar)
#     """
#     return np.min(np.max(x, axis=1))

# def margin_based(x):
#     """
#     :param x: probability matrix (n_words, n_labels)
#     :return: uncertainty measure (scalar)
#     """
#     sorted_x = np.sort(x, axis=1)
#     return np.min(sorted_x[:, -1] - sorted_x[:, -2])

# def entropy_based(x):
#     """
#     :param x: probability matrix (n_words, n_labels)
#     :return: uncertainty measure (scalar)
#     """
#     return np.max(-np.sum(x * np.log(x), axis=1))

def least_confident(x):
    """
    :param x: probability matrix (n_words, n_labels)
    :return: uncertainty measure (scalar)
    """
    return np.min(np.max(x, axis=1))

def margin_based(x):
    """
    :param x: probability matrix (n_words, n_labels)
    :return: uncertainty measure (scalar)
    """
    sorted_x = np.sort(x, axis=1)
    return np.min(sorted_x[:, -1] - sorted_x[:, -2])

def entropy_based(x):
    """
    :param x: probability matrix (n_words, n_labels)
    :return: uncertainty measure (scalar)
    """
    return np.max(-np.sum(x * np.log(x), axis=1))