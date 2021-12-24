import torch
from torch import nn
from transformers import BertTokenizer, VisualBertModel, VisualBertConfig
import numpy as np


class VisualBertClassifier(nn.Module):
    def __init__(self,
                 visual_bert_model,
                 num_classes: int = 8,
                 initial_visual_embedding_dim: int = 96,
                 final_dropout_rate: float = 0.1):
        """
        pooler_output (torch.FloatTensor of shape (batch_size, hidden_size))
        — Last layer hidden-state of the first token of the sequence (classification token)
        after further processing through the layers used for the auxiliary pretraining task.
        E.g. for BERT-family of models, this returns the classification token after processing through
        a linear layer and a tanh activation function.
         The linear layer weights are trained from the next sentence prediction (classification) objective
          during pretraining.
        @param initial_visual_embedding_dim:
        """
        super().__init__()
        self.visual_embedding_projection = nn.Linear(initial_visual_embedding_dim, 2048)
        self.visual_bert = visual_bert_model
        self.final_dropout = nn.Dropout(final_dropout_rate)
        self.out = nn.Linear(768, num_classes)

    def forward(self,
                text_input_ids,
                text_token_type_ids,
                text_attention_mask,
                visual_embeds,
                visual_token_type_ids,
                visual_attention_mask
                ):
        visual_embeds = self.visual_embedding_projection(visual_embeds)
        output = self.visual_bert(input_ids=text_input_ids,
                             token_type_ids=text_token_type_ids,
                             attention_mask=text_attention_mask,
                             visual_embeds=visual_embeds,
                             visual_token_type_ids=visual_token_type_ids,
                             visual_attention_mask=visual_attention_mask)
        output = self.final_dropout(output.pooler_output)
        output = self.out(output)
        return output


if __name__ == '__main__':
    bert_text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    inputs = bert_text_tokenizer("What is the man eating?", return_tensors="pt")

    text_input_ids = inputs.data['input_ids'].to('cuda')
    text_token_type_ids = inputs.data['token_type_ids'].to('cuda')
    text_attention_mask = inputs.data['attention_mask'].to('cuda')

    sample_face_body_embedding_path = "/home/gsoykan20/Desktop/self_development/emotion-recognition-drawings/data/emoreccom_face_body_embeddings_96d/train/0_3_4.jpg.npy"
    sample_face_body_embedding = np.load(sample_face_body_embedding_path)
    visual_embeds = torch.from_numpy(sample_face_body_embedding)
    visual_embeds = visual_embeds.to('cuda')
    visual_embeds = torch.unsqueeze(visual_embeds, 0)
    visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long).to('cuda')
    visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float).to('cuda')
    classifier = VisualBertClassifier()
    classifier.to('cuda')
    classifier.forward(text_input_ids,
                       text_token_type_ids,
                       text_attention_mask,
                       visual_embeds,
                       visual_token_type_ids,
                       visual_attention_mask)


