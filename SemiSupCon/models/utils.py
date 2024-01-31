import torch

def confusion_matrix(predictions, targets, num_classes, threshold=0.5):
    
    """
    Calculate the confusion matrix for multiclass or multilabel classification.

    Args:
    - predictions (torch.Tensor): Tensor of predicted labels (without softmax/sigmoid) of shape (B, num_classes).
    - targets (torch.Tensor): Ground truth labels (one-hot for multilabel) of shape (B, num_classes).
    - num_classes (int): Number of classes.

    Returns:
    - confusion_matrix (torch.Tensor): Confusion matrix of shape (num_classes, num_classes).
    """
    # Ensure that both predictions and targets have the same shape
    assert predictions.shape == targets.shape, "Shape mismatch between predictions and targets"

    # Threshold the predictions to obtain binary predictions for multilabel case
    if num_classes > 2:
        binary_predictions = (predictions >= threshold).int()
    else:
        binary_predictions = predictions.round().int()

    # Calculate confusion matrix for multiclass or multilabel case
    confusion_matrix = torch.matmul(targets.float().T, binary_predictions.float())

    return confusion_matrix
