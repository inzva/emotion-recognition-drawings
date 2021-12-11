from typing import Any
from pytorch_lightning import LightningModule
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import scipy.io
from sklearn.metrics import average_precision_score, precision_recall_curve
from typing import Any, List
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader 
from torchsummary import summary
from torchvision import transforms

from torch.optim.lr_scheduler import StepLR

import urllib
from src.models.modules.emotic_baseline import  FuseNetwork,DiscreteLoss,ContinuousLoss_SL1
import torchvision.models as models_torch
from src.utils.metric.micro_auc import compute_micro_auc

class VisionBaselineModel(LightningModule):
    def __init__(
        self,
        num_emotion_classes:int,
        num_cont_classes:int,
        pretrained_model_path:str,
        cat_loss_param=0.5,
        model_context_num_classes=365,
        arch:str ="resnet18",
        lr = 0.001,
        weight_decay = 5e-4,
        scheduler_step_size= 7,
        scheduler_gamma = 0.1,
        
        discrete_loss=True,
        continuous_loss = True


        ):
        super().__init__()
        self.save_hyperparameters()
        self.num_emotion_classes = num_emotion_classes
        self.arch = arch
        self.pretrained_model_path = pretrained_model_path

        self.num_cont_classes = num_cont_classes
        self.model_context = models_torch.__dict__[arch](num_classes=model_context_num_classes)
        self.places_model_dir = os.path.join(pretrained_model_path,"places")

        self.cat_loss_param = cat_loss_param

        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma= scheduler_gamma
        self.continuous = continuous_loss
        self.discrete = discrete_loss
        
        if not os.path.exists(pretrained_model_path):
            os.makedirs(pretrained_model_path)

        if  (os.path.exists(pretrained_model_path)) and "places" not in os.listdir(pretrained_model_path):
            print("Places Directory does not exist")
            print("Resnet18 trained on Places is being downloaded")
            #Places directory not exist
            
            os.makedirs(self.places_model_dir)
            
            model_url = "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"
            model_out_path =os.path.join(self.places_model_dir,"resnet18_places365.pth.tar")
            (filename, headers) = urllib.request.urlretrieve(
            model_url,
            filename = model_out_path)
            
            print("Model out path : ",model_out_path)
            checkpoint = torch.load(model_out_path, map_location=lambda storage, loc: storage) # model trained in GPU could be deployed in CPU machine like this!
            state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()} # the data parallel layer will add 'module' before each layer name
            self.model_context.load_state_dict(state_dict)
            
            torch.save(self.model_context.state_dict(), os.path.join(self.places_model_dir,'resnet18_state_dict.pth'))
            print("Downloaded file : ",filename)
            
        
        # Load Places R18 Weights for Context Model
        context_state_dict = torch.load(os.path.join(self.places_model_dir, 'resnet18_state_dict.pth'))
        self.model_context.load_state_dict(context_state_dict)

        # Body Model
        self.model_body = models_torch.resnet18(pretrained=True)
        

        self.fuse_network = FuseNetwork(list(self.model_context.children())[-1].in_features, list(self.model_body.children())[-1].in_features, \
             num_emotion_classes=self.num_emotion_classes, num_cont_class=self.num_cont_classes)
        self.model_context = nn.Sequential(*(list(self.model_context.children())[:-1]))
        self.model_body = nn.Sequential(*(list(self.model_body.children())[:-1]))

        self.metric = compute_micro_auc





        # Train only Fuse Network - Body and Context networks are freezed
        for param in self.fuse_network.parameters():
            param.requires_grad = True
        for param in self.model_context.parameters():
            param.requires_grad = True
        for param in self.model_body.parameters():
            param.requires_grad = False

        
        self.disc_loss = DiscreteLoss('dynamic', torch.device('cuda'),self.num_emotion_classes)


    
        


    def forward(self, batch):
        #print("Batch2 ",batch)
        
        img, sizes, (labels, polarities), (narrative, dialog),cropped_img = batch
        pred_context = self.model_context(img)
        pred_body = self.model_body(img)
        if self.num_cont_classes == -1:
            pred_cat = self.fuse_network(pred_context, pred_body)
            #print("Here ::: ", type(pred_cat))
            return  pred_cat
        else:

            pred_cat, pred_cont = self.fuse_network(pred_context, pred_body)
            return  pred_cat, pred_cont

         


    def step(self, batch):
        
        
        img, sizes, (labels, polarities), (narrative, dialog),cropped_img = batch
        

        if self.num_cont_classes == -1:
            pred_cat = self.forward(batch)
            # SHITTY DESIGN
            pred_cont=0

        else:
            pred_cat, pred_cont = self.forward(batch)
      
        cat_loss_batch = self.disc_loss(pred_cat, labels)
        #loss = (self.cat_loss_param * cat_loss_batch)
        loss = cat_loss_batch
        return loss, pred_cat, pred_cont, labels, labels

    def training_step(self, batch: Any, batch_idx: int):
        loss, pred_cat, pred_cont, labels_cat, labels_cont = self.step(batch)
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        micro_auc = self.metric(pred_cat, labels_cat)
        self.log("train/micro_auc", micro_auc, on_step=False, on_epoch=True, prog_bar=True)

       
        return {"loss": loss, "preds": (pred_cat, pred_cont), "targets": (labels_cat, labels_cont)}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, pred_cat, pred_cont, labels_cat, labels_cont = self.step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        micro_auc = self.metric(pred_cat, labels_cat)
        self.log("val/micro_auc", micro_auc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": (pred_cat, pred_cont), "targets": (labels_cat, labels_cont)}

    def test_step(self, batch: Any, batch_idx: int):
        loss, pred_cat, pred_cont, labels_cat, labels_cont = self.step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        micro_auc = self.metric(pred_cat, labels_cat)
        self.log("test/micro_auc", micro_auc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": (pred_cat, pred_cont), "targets": (labels_cat, labels_cont)}
    

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        opt = optim.Adam((list(self.fuse_network.parameters()) + list(self.model_context.parameters()) + \
                  list(self.model_body.parameters())), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = StepLR(opt, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma)
        return {
            "optimizer": opt,
            "lr_scheduler": scheduler
        }