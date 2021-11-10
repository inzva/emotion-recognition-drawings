import numpy as np
import torch
from sklearn import metrics


def compute_micro_auc(preds, labels):
    """
    ROC-AUC score is a good way to measure our performance for multi-class classification.
    However, it can be extrapolated to the multi-label scenario by applying it for each target separately.
    However, that will be too much for our mind to process,
    and hence, we can simply use micro AUC.
    A neat trick used in PyTorch for such multi-label classification is to use the ravel()
    function that unrolls the targets and labels,
    and then we apply the micro AUC function.

    source: https://towardsdatascience.com/multi-label-emotion-classification-with-pytorch-huggingfaces-transformers-and-w-b-for-tracking-a060d817923
    another source for understanding multiclass classification scores: https://inblog.in/AUC-ROC-score-and-curve-in-multiclass-classification-problems-2ja4jOHb2X
    @param preds:
    @param labels:
    @return:
    """
    # preds = torch.stack(preds)
    if not isinstance(preds, np.ndarray):
        preds = preds.cpu().detach().numpy()
    # labels = torch.stack(labels)
    if not isinstance(labels, np.ndarray):
        labels = labels.cpu().detach().numpy()

    '''
    ##Method 1 by taking transpose and picking each column for averaging

    auc_micro_list = []
    for i in range(n_labels):
      current_pred = preds.T[i]
      current_label = labels.T[i]
      fpr_micro, tpr_micro, _ = metrics.roc_curve(current_label.T, current_pred.T)
      auc_micro = metrics.auc(fpr_micro, tpr_micro)
      auc_micro_list.append(auc_micro)

    return {"auc": np.array(auc_micro).mean()}
    '''

    ## Method 2 using ravel()
    fpr_micro, tpr_micro, _ = metrics.roc_curve(labels.ravel(), preds.ravel())
    auc_micro = metrics.auc(fpr_micro, tpr_micro)
    return auc_micro
