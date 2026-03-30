import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0
    
    pred = prediction.astype(int)
    truth = ground_truth.astype(int)
    
    FN = np.sum((pred == 0) & (truth == 1))
    FP = np.sum((pred == 1) & (truth == 0))
    TN = np.sum((pred == 0) & (truth == 0))
    TP = np.sum((pred == 1) & (truth == 1))
    
    # accuracy
    accuracy = (TN + TP) / (TN + TP + FN + FP)
    
    # precision
    if TP + FP == 0:
        precision = 0.0
    else:
        precision = TP / (TP + FP)
    
    # recall
    if TP + FN == 0:
        recall = 0.0
    else:
        recall = TP / (TP + FN)
           
    # f1
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    
    correct_prediction = np.sum(prediction == ground_truth)
    accuracy = correct_prediction / len(prediction)
    return accuracy
