import wandb
import numpy as np
import torch
import pickle

import torch.nn.functional as F
import matplotlib.pyplot as plt

from collections import namedtuple
from types import SimpleNamespace
from typing import List, Tuple

from .helpers import DataLoader, DirHelper, Batcher
from .utils.torch_utils import no_grad
from .model import TransformerModel

class Trainer():
    """"base class for running basic transformer classification models"""
    
    def __init__(self, exp_name:str, m_args:namedtuple):
        self.dir = DirHelper(exp_name, m_args.temp)
        self.dir.save_args('model_args.json', m_args)
        self.set_up_helpers(m_args)
        
    ############  MAIN TRAIN LOOP  #################################
    
    def set_up_helpers(self, m_args:namedtuple):
        self.model_args = m_args
        self.data_loader = DataLoader(m_args.transformer, formatting=m_args.formatting)
        self.batcher = Batcher(max_len=m_args.max_len)
        self.model = TransformerModel(trans_name=m_args.transformer)
        self.device = m_args.device

    def train(self, t_args:namedtuple):
        self.dir.save_args('train_args.json', t_args)
        if t_args.wandb: self.set_up_wandb(t_args)
 
        train, dev, test = self.data_loader(t_args.data_set, t_args.lim)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=t_args.lr)
        best_epoch = (-1, 10000, 0)
        self.to(self.device)
        
        for epoch in range(t_args.epochs):
            ######  TRAINING  ##############################
            self.model.train()
            self.dir.reset_metrics()
            train_b = self.batcher(data=train, bsz=t_args.bsz, shuffle=True)
            
            for k, batch in enumerate(train_b, start=1):
                output = self.model_output(batch)

                optimizer.zero_grad()
                output.loss.backward()
                optimizer.step()

                # accuracy logging
                self.dir.update_avg_metrics(loss=output.loss)
                self.dir.update_acc_metrics(hits=output.hits, 
                                            num_preds=output.num_preds)
                
                # print train performance every now and then
                if k%t_args.print_len == 0:
                    perf = self.dir.print_perf('train', epoch, k)
                    if t_args.wandb:
                         wandb.log({'epoch':epoch, 'loss':perf.loss, 'acc':perf.acc})
            
            ######  DEV  ##################################
            self.model.eval()
            perf = self.system_eval(dev, epoch, mode='dev')
            if t_args.wandb:
                wandb.log({"dev_loss":perf.loss, "dev_acc":perf.acc})

            # save performance if best dev performance 
            if perf.acc < best_epoch[1]:
                best_epoch = (epoch, perf.loss, perf.acc)
                if t_args.save: self.save_model()

            ######  TEST  #################################
            perf = self.system_eval(test, epoch, mode='test')
        
        print(f'best dev epoch: {best_epoch}')

    def model_output(self, batch):
        y = self.model(input_ids=batch.ids, 
                       attention_mask=batch.mask)
                
        loss = F.cross_entropy(y, batch.labels)
            
        # return accuracy metrics
        hits = torch.argmax(y, dim=-1) == batch.labels
        hits = torch.sum(hits[batch.labels != -100]).item()
        num_preds = torch.sum(batch.labels != -100).item()
                
        return SimpleNamespace(loss=loss, y=y,
                               hits=hits, num_preds=num_preds)
    
    
    ############# EVAL METHODS ####################################
    @no_grad
    def system_eval(self, data, epoch:int, mode='dev'):
        self.dir.reset_metrics()         
        batches = self.batcher(data=data, bsz=1, shuffle=False)
        for k, batch in enumerate(batches, start=1):
            output = self.model_output(batch)
            self.dir.update_avg_metrics(loss=output.loss)
            self.dir.update_acc_metrics(hits=output.hits, 
                                        num_preds=output.num_preds)
        perf = self.dir.print_perf(mode, epoch, 0)
        return perf    

    #############  MODEL UTILS  ###################################
    
    def save_model(self, name:str='base'):
        device = next(self.model.parameters()).device
        self.model.to("cpu")
        torch.save(self.model.state_dict(), 
                   f'{self.dir.abs_path}/models/{name}.pt')
        self.model.to(self.device)

    def load_model(self, name:str='base'):
        self.model.load_state_dict(
            torch.load(self.dir.abs_path + f'/models/{name}.pt'))

    def to(self, device):
        assert hasattr(self, 'model') and hasattr(self, 'batcher')
        self.model.to(device)
        self.batcher.to(device)

    ############  WANDB UTILS  ####################################
    
    def set_up_wandb(self, args:namedtuple):
        wandb.init(project=args.wandb, entity="adian", reinit=True,
                   name=self.dir.exp_name, dir=self.dir.abs_path)

        # save experiment config details
        cfg = {}
        cfg['epochs']      = args.epochs
        cfg['bsz']         = args.bsz
        cfg['lr']          = args.lr
        cfg['transformer'] = self.model_args.transformer
        wandb.config.update(cfg) 
        wandb.watch(self.model)
        
    
