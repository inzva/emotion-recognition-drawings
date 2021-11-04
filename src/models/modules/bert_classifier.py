from torch import nn


# not needed arguments
# n_train_steps
# self.step_scheduler_after = "batch"

class BertClassifier(nn.Module):
    def __init__(self,
                 bert_model,
                 num_classes: int,
                 dropout_rate: float = 0.2
                 ):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(768, num_classes)

    def forward(self, ids, mask):
        output_1 = self.bert(ids, attention_mask=mask)["pooler_output"]
        output_2 = self.dropout(output_1)
        output = self.out(output_2)
        return output
