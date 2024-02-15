
from torchmetrics.functional import auroc, average_precision, accuracy, recall, precision
from mir_eval.key import weighted_score
import torch

## all metric functions return a dictionary with the metrics

def multilabel_metrics(logits, labels, idx2class, class_names, n_classes):
    preds = torch.sigmoid(logits)
    aurocs = auroc(preds,labels,task = 'multilabel',num_labels = n_classes)
    ap_score = average_precision(preds,labels,task = 'multilabel',num_labels = n_classes)
    return {'auroc':aurocs,'ap':ap_score}

def mtat_top50_metrics(logits, labels, idx2class, class_names, n_classes):
    return multilabel_metrics(logits, labels, idx2class, class_names, n_classes)

def mtat_all_metrics(logits, labels, idx2class, class_names, n_classes):
    return multilabel_metrics(logits, labels, idx2class, class_names, n_classes)

def mtg_top50_metrics(logits, labels, idx2class, class_names, n_classes):
    return multilabel_metrics(logits, labels, idx2class, class_names, n_classes)

def mtg_genre_metrics(logits, labels, idx2class, class_names, n_classes):
    return multilabel_metrics(logits, labels, idx2class, class_names, n_classes)

def mtg_instr_metrics(logits, labels, idx2class, class_names, n_classes):
    return multilabel_metrics(logits, labels, idx2class, class_names, n_classes)

def mtg_mood_metrics(logits, labels, idx2class, class_names, n_classes):
    return multilabel_metrics(logits, labels, idx2class, class_names, n_classes)
    
def giantsteps_metrics(logits, labels, idx2class, class_names, n_classes):
    preds = torch.softmax(logits,dim = 1)
    preds = torch.argmax(preds,dim = 1)
    batch_size = preds.size(0)
    preds_names = [idx2class[pred] for pred in preds.cpu().numpy()]
    labels_idx = torch.argmax(labels,dim = 1)
    labels_names = [idx2class[label] for label in labels_idx.cpu().numpy()]
    accuracy_ = accuracy(preds,labels_idx, task = 'multiclass', num_classes = n_classes)
    weighted_score_ = 0
    
    for i in range(batch_size):
        weighted_score_ += weighted_score(reference_key = labels_names[i], estimated_key = preds_names[i])
    weighted_score_ = weighted_score_/batch_size
    
    return {'accuracy':accuracy_,'weighted_score':weighted_score_}


def nsynth_pitch_metrics(logits, labels, idx2class, class_names, n_classes):
    softlogits = torch.softmax(logits,dim = 1)
    preds = torch.argmax(softlogits,dim = 1)
    batch_size = preds.size(0)
    preds_names = [idx2class[pred] for pred in preds.cpu().numpy()]
    labels_idx = torch.argmax(labels,dim = 1)
    labels_names = [idx2class[label] for label in labels_idx.cpu().numpy()]
    accuracy_ = accuracy(softlogits,labels_idx, task = 'multiclass', num_classes = n_classes)
    recall_ = recall(preds,labels_idx,task = 'multiclass',num_classes = n_classes)
    precision__ = precision(preds,labels_idx,task = 'multiclass',num_classes = n_classes)
    
    return {'accuracy':accuracy_,'recall':recall_,'precision':precision__}

def nsynth_instr_family_metrics(logits, labels, idx2class, class_names, n_classes):
    softlogits = torch.softmax(logits,dim = 1)
    preds = torch.argmax(softlogits,dim = 1)
    batch_size = preds.size(0)
    preds_names = [idx2class[pred] for pred in preds.cpu().numpy()]
    labels_idx = torch.argmax(labels,dim = 1)
    labels_names = [idx2class[label] for label in labels_idx.cpu().numpy()]
    accuracy_ = accuracy(softlogits,labels_idx, task = 'multiclass', num_classes = n_classes)
    recall_ = recall(preds,labels_idx,task = 'multiclass',num_classes = n_classes)
    precision__ = precision(preds,labels_idx,task = 'multiclass',num_classes = n_classes)
    
    return {'accuracy':accuracy_,'recall':recall_,'precision':precision__}

def gtzan_metrics(logits, labels, idx2class, class_names, n_classes):
    preds = torch.softmax(logits,dim = 1)
    preds = torch.argmax(preds,dim = 1)
    batch_size = preds.size(0)
    preds_names = [idx2class[pred] for pred in preds.cpu().numpy()]
    labels_idx = torch.argmax(labels,dim = 1)
    labels_names = [idx2class[label] for label in labels_idx.cpu().numpy()]
    accuracy_ = accuracy(preds,labels_idx, task = 'multiclass', num_classes = n_classes)
    
    return {'accuracy':accuracy_}