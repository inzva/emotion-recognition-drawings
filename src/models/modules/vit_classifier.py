from torch import nn
from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTModel
from torch import Tensor

from PIL import Image
import requests


class ViTClassifier(nn.Module):
    def __init__(self,
                 feature_extractor_alias: str = "google/vit-base-patch16-224",
                 model_alias: str = "google/vit-base-patch16-224-in21k",
                 num_classes: int = 8,
                 use_feature_extractor: bool = False):
        """
        sources:
         - https://github.com/NielsRogge/Transformers-Tutorials/tree/master/VisionTransformer
         - https://huggingface.co/docs/transformers/model_doc/vit
         - https://huggingface.co/google/vit-base-patch16-224
        @param feature_extractor_alias:
        @param model_alias:
        @param num_classes: number of classes for classification head
        @param use_feature_extractor: use it for single image processing
        """
        super().__init__()
        self.use_feature_extractor = use_feature_extractor
        if use_feature_extractor:
            self.feature_extractor = ViTFeatureExtractor.from_pretrained(feature_extractor_alias)
        self.model = ViTModel.from_pretrained(model_alias)
        self.emb_size = self.model.config.hidden_size
        self.classification_head = nn.Linear(self.emb_size, num_classes)

    def forward(self, inputs):
        if self.use_feature_extractor:
            inputs = self.feature_extractor(images=inputs, return_tensors="pt")
            outputs = self.model(**inputs)
        else:
            outputs = self.model(inputs)
        last_hidden_states = outputs.last_hidden_state
        cls_hidden_state = last_hidden_states[:, 0, :]
        cls_logits = self.classification_head(cls_hidden_state)
        return cls_logits


if __name__ == '__main__':
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    classifier = ViTClassifier().to('cpu')
    classifier.forward([image])
