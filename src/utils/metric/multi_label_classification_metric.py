import enum
from torchmetrics.classification.auroc import AUROC
from src.utils.metric.micro_auc import compute_micro_auc
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auc_scikit(preds,
                       labels,
                       multi_class='ovr',
                       average='macro'):
    """
    Error1: Good source to explain why "ValueError: multilabel-indicator format is not supported" is raised
    source: https://stackoverflow.com/a/65243791/8265079
    so this can not be used.
        fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)
        return metrics.auc(fpr, tpr)
    Error2: good source to explain why ValueError("Only one class present in y_true. ROC AUC score "
    source: https://stackoverflow.com/a/45139405/8265079
    The basic reason is class inbalance, especially for "Other" class.
    Because there are cases where all labels are 0 in whole batch.
    @param preds:
    @param labels:
    @param multi_class: {‘raise’, ‘ovr’, ‘ovo’}, default=’raise’
    Only used for multiclass targets. Determines the type of configuration to use. The default value raises an error, so either 'ovr' or 'ovo' must be passed explicitly.
    'ovr':
        Stands for One-vs-rest. Computes the AUC of each class against the rest [3] [4]. This treats the multiclass case in the same way as the multilabel case. Sensitive to class imbalance even when average == 'macro', because class imbalance affects the composition of each of the ‘rest’ groupings.
    'ovo':
        Stands for One-vs-one. Computes the average AUC of all possible pairwise combinations of classes [5]. Insensitive to class imbalance when average == 'macro'.
    @param average: {'macro', 'micro'}
    'micro':
    Calculate metrics globally by considering each element of the label indicator matrix as a label.
    'macro':
    Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    'weighted':
    Calculate metrics for each label, and find their average, weighted by support (the number of true instances for each label).
    @return:
    """
    if not isinstance(preds, np.ndarray):
        preds = preds.cpu().detach().numpy()
    if not isinstance(labels, np.ndarray):
        labels = labels.cpu().detach().numpy()
    try:
        score = roc_auc_score(labels,
                              preds,
                              multi_class=multi_class,
                              average=average)
        return score
    except ValueError:
        return 0


class MultiLabelClassificationMetric(enum.Enum):
    RocCurve_Scikit = 1
    AUROC_Torchmetrics = 2
    MicroRocCurve_Scikit = 3

    @classmethod
    def get_implementation(cls, metric, num_classes: int = 8):
        if metric == MultiLabelClassificationMetric.MicroRocCurve_Scikit:
            return compute_micro_auc
        elif metric == MultiLabelClassificationMetric.RocCurve_Scikit:
            return compute_auc_scikit
        elif metric == MultiLabelClassificationMetric.AUROC_Torchmetrics:
            return AUROC(pos_label=1,
                         num_classes=num_classes,
                         average='micro')
        else:
            raise Exception(
                "Unknown MultiLabelClassificationMetric to get implementation of"
            )
