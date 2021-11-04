import torch
from torch.utils.data import Dataset


class GoEmotions(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        """ GoEmotions dataset, source: https://towardsdatascience.com/multi-label-emotion-classification-with-pytorch-huggingfaces-transformers-and-w-b-for-tracking-a060d817923
        GoEmotions is a corpus of 58k carefully curated comments extracted from Reddit, with human annotations to 27 emotion categories or Neutral.
        link: https://github.com/google-research/google-research/tree/master/goemotions
        @param texts:
        @param labels:
        @param tokenizer: tokenizer for the dataset, this is decided to be
        an argument because it would provide ease to try different tokenizers
        @param max_len:
        """
        self.texts = texts
        self.labels = labels

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        inputs = self.tokenizer.__call__(text,
                                         None,
                                         add_special_tokens=True,
                                         max_length=self.max_len,
                                         padding="max_length",
                                         truncation=True,
                                         )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long)
        }
