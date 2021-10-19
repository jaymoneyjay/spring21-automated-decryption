""" Defines functions to be used in the project. """
from sklearn.metrics import confusion_matrix


def normed_confusion(y_pred, y_truth):
    """ Compute confusion matrix normed over samples """
    
    return confusion_matrix(y_pred, y_truth, normalize="true")

def hamming_distance(string1, string2):
    """ Compute the hamming distance between two equally sized strings"""
    
    assert len(string1) == len(string2), "Strings must have the same length"
    return sum(s1 != s2 for s1, s2 in zip(string1, string2))

def hamming_similarity(string1, string2):
    """ Compute similarity of two strings based on hamming distance """
    
    return 1 - hamming_distance(string1, string2) / len(string1)


