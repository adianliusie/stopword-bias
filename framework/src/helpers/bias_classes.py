import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from types  import SimpleNamespace
from typing import List

from .data_loader import DataLoader
from .batcher     import Batcher

class BiasDataLoader(DataLoader):
    def __init__(self, trans_name:str, formatting:str, bias_model:'Trainer'):
        super().__init__(trans_name, formatting)

        #load SystemLoader here to avoid circular imports
        from ..system_loader import SystemLoader, EnsembleLoader

        #set up bias model
        self.bias_model = EnsembleLoader(bias_model)
        self.get_biased_preds('rt')

    def get_data(self, data_name:str, lim:int=None):
        train, dev, test = super().get_data(data_name, lim)
        train_preds, dev_preds, test_preds = self.get_biased_preds(data_name)
        train = self.augment_split(train, train_preds)
        dev   = self.augment_split(dev, dev_preds)
        test  = self.augment_split(test, test_preds)
        return train, dev, test

    def get_biased_preds(self, data_name:str):
        train = self.bias_model.load_probs(data_name, 'train')
        dev   = self.bias_model.load_probs(data_name, 'dev')
        test   = self.bias_model.load_probs(data_name, 'test')
        train, dev, test = [self.convert_preds_logit(i) for i in (train, dev, test)]
        return train, dev, test

    @staticmethod
    def convert_preds_logit(preds:dict)->dict:
        output = {k:np.log(v) for k, v in preds.items()}
        return output

    @staticmethod
    def augment_split(split:List[SimpleNamespace], bias_preds:dict):
        output = []
        for k, ex in enumerate(split):
            bias_pred = bias_preds[k]
            aug_ex = {'text':ex.text, 'ids':ex.ids, 
                      'bias_pred':bias_pred, 'label':ex.label}
            output.append(SimpleNamespace(**aug_ex))
        return output

class BiasBatcher(Batcher):
    def batchify(self, batch:List[list]):
        """each input is input ids and mask for utt, + label"""
        sample_id, ids, bias_preds, labels = zip(*batch)  
        ids, mask = self._get_padded_ids(ids)
        bias_preds = torch.FloatTensor(bias_preds).to(self.device)
        labels = torch.LongTensor(labels).to(self.device)
        return SimpleNamespace(sample_id=sample_id, ids=ids, mask=mask, 
                               bias_preds=bias_preds, labels=labels)
    
    def _prep_examples(self, data:list):
        """ sequence classification input data preparation"""
        prepped_examples = []
        for k, ex in enumerate(data):
            ids   = ex.ids
            label = ex.label
            bias_pred = ex.bias_pred
            
            if len(ids) > self.max_len:            
                ids = ids[:self.max_len-1] + [ids[-1]]
            prepped_examples.append([k, ids, bias_pred, label])       
        return prepped_examples
        
class LearnedMixin(torch.nn.Module):
    def __init__(self, penalty:float=0.03):
        super().__init__()
        self.penalty = penalty
        self.bias_lin = torch.nn.Linear(768, 1)

    def forward(self, hidden, logits, bias, labels):
        logits = logits.float()  # In case we were in fp16 mode
        logits = F.log_softmax(logits, 1)

        factor = self.bias_lin.forward(hidden)
        factor = factor.float()
        factor = F.softplus(factor)

        bias = bias * factor

        bias_lp = F.log_softmax(bias, 1)
        entropy = -(torch.exp(bias_lp) * bias_lp).sum(1).mean(0)

        loss = F.cross_entropy(logits + bias, labels) + self.penalty*entropy
        return SimpleNamespace(loss=loss, y=logits + bias)  
