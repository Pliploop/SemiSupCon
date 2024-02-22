
from torchmetrics.functional import auroc, average_precision, accuracy, recall, precision, r2_score
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
    return get_multiclass_metrics(logits, labels, idx2class, class_names, n_classes, 'nsynth_pitch')

def nsynth_instr_family_metrics(logits, labels, idx2class, class_names, n_classes):
    return get_multiclass_metrics(logits, labels, idx2class, class_names, n_classes, 'nsynth_instr_family')

def gtzan_metrics(logits, labels, idx2class, class_names, n_classes):
    return get_multiclass_metrics(logits, labels, idx2class, class_names, n_classes, 'gtzan')

def vocalset_technique_metrics(logits, labels, idx2class, class_names, n_classes):
    return get_multiclass_metrics(logits, labels, idx2class, class_names, n_classes, 'vocalset_technique')

def vocalset_singer_metrics(logits, labels, idx2class, class_names, n_classes):
    return get_multiclass_metrics(logits, labels, idx2class, class_names, n_classes, 'vocalset_language')

def medleydb_metrics(logits, labels, idx2class, class_names, n_classes):
    return get_multiclass_metrics(logits, labels, idx2class, class_names, n_classes, 'medleydb')

def emomusic_metrics(logits, labels,idx2class, class_names, n_classes):
    
    global_r2 = r2_score(preds = logits, target = labels, multioutput = 'uniform_average')
    v_r2 = r2_score(preds = logits[:,1], target = labels[:,1])
    a_r2 = r2_score(preds = logits[:,0], target = labels[:,0])
    
    return {
        'r2_score':global_r2,
        'valence_r2_score':v_r2,
        'arousal_r2_score':a_r2
    }
    
def get_multiclass_metrics(logits, labels, idx2class, class_names, n_classes, dataset):
    preds = torch.softmax(logits,dim = 1)
    preds = torch.argmax(preds,dim = 1)
    batch_size = preds.size(0)
    preds_names = [idx2class[pred] for pred in preds.cpu().numpy()]
    labels_idx = torch.argmax(labels,dim = 1)
    labels_names = [idx2class[label] for label in labels_idx.cpu().numpy()]
    accuracy_ = accuracy(preds,labels_idx, task = 'multiclass', num_classes = n_classes)
    precision_ = precision(preds,labels_idx, task = 'multiclass', num_classes = n_classes)
    recall_ = recall(preds,labels_idx, task = 'multiclass', num_classes = n_classes)
    
    return {'accuracy':accuracy_,'precision':precision_,'recall':recall_}