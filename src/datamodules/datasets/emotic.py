import ast
import numpy as np 
import os 
from PIL import Image

import torch 
from torch.utils.data import Dataset
from torchvision import transforms


class EmoticDataset(Dataset):
  ''' 
  Custom Emotic dataset class. Use preprocessed data stored in npy files.
  Emotic Pytorch Implementation : https://github.com/Tandon-A/emotic  
  Original Dataset for Emotic Dataset Access : https://github.com/rkosti/emotic
  
   '''
  def __init__(self, x_context, x_body, y_cat, y_cont, transform, context_norm, body_norm):
    super(EmoticDataset,self).__init__()
    self.x_context = x_context
    self.x_body = x_body
    self.y_cat = y_cat 
    self.y_cont = y_cont
    self.transform = transform 
    self.context_norm = transforms.Normalize(context_norm[0], context_norm[1])  # Normalizing the context image with context mean and context std
    self.body_norm = transforms.Normalize(body_norm[0], body_norm[1])           # Normalizing the body image with body mean and body std

  def __len__(self):
    return len(self.y_cat)
  
  def __getitem__(self, index):
    image_context = self.x_context[index]
    image_body = self.x_body[index]
    cat_label = self.y_cat[index]
    cont_label = self.y_cont[index]
    return self.context_norm(self.transform(image_context)), self.body_norm(self.transform(image_body)), torch.tensor(cat_label, dtype=torch.float32), torch.tensor(cont_label, dtype=torch.float32)/10.0


  def cat_to_one_hot(self, cat):
    one_hot_cat = np.zeros(26)
    for em in cat:
      one_hot_cat[self.cat2ind[em]] = 1
    return one_hot_cat