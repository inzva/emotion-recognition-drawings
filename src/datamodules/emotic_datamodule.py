from typing import Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
from scipy.io import loadmat 

import os
from src.utils.mat2py import prepare_data
import numpy as np
from src.datamodules.datasets.emotic import EmoticDataset
from torchvision import transforms

class EmoticDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 save_path ="../data/emotic_pre",
                 batch_size: int = 64,
                 num_workers: int = 0,
                 pin_memory: bool = False
                 ):
        super().__init__()
        
        self.data_dir = data_dir
        self.save_path = save_path 
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data = None

        self.emotion_cats = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection',
       'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear',
       'Happiness', 'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']
        self.cat2ind = {}
        self.ind2cat = {}
        for idx, emotion in enumerate(self.emotion_cats):
            self.cat2ind[emotion] = idx
            self.ind2cat[idx] = emotion


    @property
    def num_classes(self) -> int:
        return len(self.emotion_cats)  # 26 Emotion Classes

    def prepare_data(self) -> None:
        # Reads Annotation.mat file preprocess and save to the folder emotic_pre
        print(os.listdir('..'))
        if "emotic_pre" not in os.listdir(self.data_dir):
            os.makedirs(os.path.join(self.data_dir,"emotic_pre"))
            print ('Emotic Pre is not exist : numpy files are being created')
            
            ann_path_src = os.path.join(self.data_dir, 'Annotations','Annotations.mat')
            mat = loadmat(ann_path_src)
            labels = ['train', 'val', 'test']
            for label in labels:
                data_mat = mat[label]
                print ('starting label ', label)
                data_path = os.path.join(self.data_dir, 'emotic')
                save_path = os.path.join(self.data_dir, 'emotic_pre')
                prepare_data(data_mat, data_path ,  save_path,self.cat2ind, self.ind2cat, dataset_type=label, generate_npy=True, debug_mode=False)

        else:
            print("Emotic_pre files are already exist Skipping")
        
            

    def setup(self, stage: Optional[str] = None):

        # After having preprocessed np files
    
        # Load training preprocessed data
        train_context = np.load(os.path.join(self.data_dir,'emotic_pre','train_context_arr.npy'))
        train_body = np.load(os.path.join(self.data_dir,'emotic_pre','train_body_arr.npy'))
        train_cat = np.load(os.path.join(self.data_dir,'emotic_pre','train_cat_arr.npy'))
        train_cont = np.load(os.path.join(self.data_dir,'emotic_pre','train_cont_arr.npy'))

        # Load validation preprocessed data 
        val_context = np.load(os.path.join(self.data_dir,'emotic_pre','val_context_arr.npy'))
        val_body = np.load(os.path.join(self.data_dir,'emotic_pre','val_body_arr.npy'))
        val_cat = np.load(os.path.join(self.data_dir,'emotic_pre','val_cat_arr.npy'))
        val_cont = np.load(os.path.join(self.data_dir,'emotic_pre','val_cont_arr.npy'))

        # Load testing preprocessed data
        test_context = np.load(os.path.join(self.data_dir,'emotic_pre','test_context_arr.npy'))
        test_body = np.load(os.path.join(self.data_dir,'emotic_pre','test_body_arr.npy'))
        test_cat = np.load(os.path.join(self.data_dir,'emotic_pre','test_cat_arr.npy'))
        test_cont = np.load(os.path.join(self.data_dir,'emotic_pre','test_cont_arr.npy'))


        print ('train ', 'context ', train_context.shape, 'body', train_body.shape, 'cat ', train_cat.shape, 'cont', train_cont.shape)
        print ('val ', 'context ', val_context.shape, 'body', val_body.shape, 'cat ', val_cat.shape, 'cont', val_cont.shape)
        print ('test ', 'context ', test_context.shape, 'body', test_body.shape, 'cat ', test_cat.shape, 'cont', test_cont.shape)
                            

        context_mean = [0.4690646, 0.4407227, 0.40508908]
        context_std = [0.2514227, 0.24312855, 0.24266963]
        body_mean = [0.43832874, 0.3964344, 0.3706214]
        body_std = [0.24784276, 0.23621225, 0.2323653]
        context_norm = [context_mean, context_std]
        body_norm = [body_mean, body_std]


        train_transform = transforms.Compose([transforms.ToPILImage(), 
                                            transforms.RandomHorizontalFlip(), 
                                            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), 
                                            transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToPILImage(), 
                                            transforms.ToTensor()])

        self.data_train = EmoticDataset(train_context, train_body, train_cat, train_cont, \
                                  train_transform, context_norm, body_norm)
        self.data_val = EmoticDataset(val_context, val_body, val_cat, val_cont, \
                                        test_transform, context_norm, body_norm)
        self.data_test = EmoticDataset(test_context, test_body, test_cat, test_cont, \
                                        test_transform, context_norm, body_norm)


        

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )


if __name__ == '__main__':
    datamodule = EmoticDataModule()
    datamodule.prepare_data()
    datamodule.setup()
    train_dataloader = iter(datamodule.train_dataloader())
    batch = next(train_dataloader)  # img, img_info, (hard labels, polarities), text encodings
    print(batch)
