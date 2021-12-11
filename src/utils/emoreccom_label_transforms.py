import numpy as np


def normalize_and_take_top_n(emoreccom_labels,
                             polarity_threshold: float = 0.2,
                             take_top_n: int = 1):
    labels, polarity = emoreccom_labels
    # normalize
    polarity = np.divide(polarity, np.sum(polarity))
    # create mask
    polarity_mask = polarity > polarity_threshold
    labels = labels * polarity_mask
    polarity *= polarity_mask
    top_indices = np.argpartition(polarity, -take_top_n, axis=0)[-take_top_n:]
    top_index_mask = np.zeros_like(labels)
    for idx, top_index in enumerate(top_indices):
        top_index_mask[top_index] = 1
    labels *= top_index_mask
    return labels, polarity
