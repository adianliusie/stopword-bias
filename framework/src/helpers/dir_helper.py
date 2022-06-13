import json
import pickle
import os
import shutil
import numpy as np

from typing import Callable
from types import SimpleNamespace
from collections import namedtuple, OrderedDict
from scipy.stats import pearsonr, spearmanr

from ..utils.json_utils import load_json, save_json

class DirHelper():
    def __init__(self, exp_path:str=None, temp:bool=False):
        if temp:
            print("using temp directory")
            self.exp_path = 'trained_models/temp'
            self.del_temp_dir()
        else:
            self.exp_path = exp_path
        
        self.abs_path = os.path.abspath(self.exp_path)
        self.set_up_dir()
        self.log = self.make_logger(file_name='log')

    def set_up_dir(self):
        """makes experiments directory"""
        os.makedirs(self.abs_path)
        os.mkdir(f'{self.abs_path}/models')
        os.mkdir(f'{self.abs_path}/preds')

    def make_logger(self, file_name:str)->Callable:
        """creates logging function which saves prints to txt file"""
        log_path = f'{self.abs_path}/{file_name}.txt'
        open(log_path, 'a+').close()  
        
        def log(*x):
            print(*x)    
            with open(log_path, 'a') as f:
                for i in x:
                    f.write(str(i) + ' ')
                f.write('\n')
        return log
    
    @property
    def exp_name(self):
        return self.exp_path.replace('trained_models/','')
    
    def del_temp_dir(self):
        """deletes the temp, unsaved experiments directory"""
        if os.path.isdir('trained_models/temp'): 
            shutil.rmtree('trained_models/temp')        

    ### METHODS FOR LOGGING ############################################
    
    def reset_metrics(self):
        self.metrics = OrderedDict()
        self.acc = np.zeros(2)
        self.preds, self.labels = [], []
        self.samples = 0

    def update_avg_metrics(self, **kwargs):
        for key, val in kwargs.items():
            if key not in self.metrics: self.metrics[key]=0
            self.metrics[key] += val
        self.samples += 1
   
    def update_acc_metrics(self, hits:int=0, num_preds:int=0):
        self.acc += [hits, num_preds]
    
    def update_preds(self, y:int, label:int):
        self.preds.append(y)
        self.labels.append(label)
        
    def print_perf(self, mode:str, epoch:int, step:int):
        """returns and logs performance"""
        metrics = OrderedDict([(k, v/self.samples) for k, v in self.metrics.items()])
        acc  = f'{self.acc[0]/self.acc[1]:.3f}' if self.acc[0]>0 else 0
        metrics_str = [f'{k} {v:6.3f}' for k, v in metrics.items()]

        # logging performance
        step_print = step if mode=='train' else mode
        metrics_str = '  '.join(metrics_str)
        self.log(f'{epoch:<3} {step_print:<5}  acc {acc}  {metrics_str}')
        
        self.reset_metrics()                        
        return SimpleNamespace(acc=float(acc), **metrics)
    
    def print_reg_perf(self, mode:str, epoch:int, step:int):
        metrics = OrderedDict([(k, v/self.samples) for k, v in self.metrics.items()])
        metrics['RMSE']  = metrics['loss']**0.5
        metrics['pear']  = pearsonr(self.preds, self.labels)[0]
        metrics['spear'] = spearmanr(self.preds, self.labels)[0]
        metrics_str = [f'{k} {v:6.3f}' for k, v in metrics.items()]

        # logging performance
        step_print = step if mode=='train' else mode
        metrics_str = '  '.join(metrics_str)
        self.log(f'{epoch:<3}  {step_print:<5} {metrics_str:<5}')
        
        self.reset_metrics()                        
        return SimpleNamespace(**metrics)
    
    ### Utility Methods ################################################
    
    def file_exists(self, file_name:str)->bool:
        return os.path.isfile(f'{self.abs_path}/{file_name}') 
    
    def probs_exists(self, data_name:str, mode:str, dir_name='preds')->bool:
        eval_name = f'{data_name}_{mode}'
        pred_path = f'{self.abs_path}/{dir_name}/{eval_name}'
        return os.path.isfile(pred_path) 

    def save_args(self, name:str, data:namedtuple):
        """saves arguments into json format"""
        save_path = f'{self.abs_path}/{name}'
        save_json(data.__dict__, save_path)

    def save_dict(self, name:str, data:dict):
        save_path = f'{self.abs_path}/{name}'
        save_json(data, save_path)

    def save_probs(self, preds, data_name, mode, dir_name='preds'):
        """saves predictions directory"""
        eval_name = f'{data_name}_{mode}'
        pred_path = f'{self.abs_path}/{dir_name}/{eval_name}'
        
        with open(pred_path, 'wb') as handle:
            pickle.dump(preds, handle)
    
    def make_dir(self, dir_name:str):
        if not os.path.isdir(f'{self.abs_path}/{dir_name}'): 
            print(f'{self.abs_path}/{dir_name}')
            os.mkdir(f'{self.abs_path}/{dir_name}')
        
    ### Methods for loading ############################################
  
    @classmethod
    def load_dir(cls, exp_path:str)->'DirHelper':
        dir_manager = cls.__new__(cls)
        dir_manager.exp_path = exp_path
        dir_manager.abs_path = os.path.abspath(exp_path)
        dir_manager.log = print
        return dir_manager
    
    def load_args(self, name:str)->SimpleNamespace:
        args = load_json(f'{self.abs_path}/{name}')
        return SimpleNamespace(**args)
    
    def load_dict(self, name:str)->dict:
        return load_json(f'{self.abs_path}/{name}')
    
    def load_probs(self, data_name:str, mode:str, dir_name='preds'):
        """saves predictions directory"""
        eval_name = f'{data_name}_{mode}'
        pred_path = f'{self.abs_path}/{dir_name}/{eval_name}'
        with open(pred_path, 'rb') as handle:
            predictions = pickle.load(handle)
        return predictions
    