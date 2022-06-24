import torch.nn.functional as F
import numpy as np
import os

from tqdm import tqdm

from .trainer import Trainer
from .utils.torch_utils import no_grad
from .utils.data_utils import load_data
from .helpers import DirHelper
        
class SystemLoader(Trainer):
    """Base loader class- the inherited class inherits
       the Trainer so has all experiment methods"""

    def __init__(self, exp_path:str):
        self.dir = DirHelper.load_dir(exp_path)
        
    def set_up_helpers(self):
        #load training arguments and set up helpers
        args = self.dir.load_args('model_args.json')
        super().set_up_helpers(args)

        #load final model
        self.load_model()
        self.model.eval()
        self.device = 'cuda:0'
        self.to(self.device)

    def load_preds(self, data_name, mode):
        probs = self.load_probs(data_name, mode)
        preds = {}
        for k, probs in probs.items():
            preds[k] = int(np.argmax(probs, axis=-1))  
        return preds
        
    def load_probs(self, data_name, mode):
        """loads predictions if saved, else generates"""
        if not self.dir.probs_exists(data_name, mode):
            self.set_up_helpers()
            self.generate_probs(data_name, mode)
        probs = self.dir.load_probs(data_name, mode)
        return probs

    def generate_probs(self, data_name, mode):
        probabilties = self._probs(data_name, mode)
        self.dir.save_probs(probabilties, data_name, mode)

    @no_grad
    def _probs(self, data_name, mode='test'):
        """get model predictions for given data"""
        self.model.eval()
        self.to(self.device)
        eval_batches = self._get_eval_batches(data_name, mode)

        probabilties = {}
        for batch in tqdm(eval_batches):
            sample_id = batch.sample_id[0]
            output = self.model_output(batch)

            y = output.y.squeeze(0)
            if y.shape and y.shape[-1] > 1:  # Get probabilities of predictions
                y = F.softmax(y, dim=-1)
            probabilties[sample_id] = y.cpu().numpy()
        return probabilties
    
    def _get_eval_batches(self, data_name, mode='test'):
        #get eval data- data_loader returns (train, dev, test) so index
        eval_data = self.data_loader.get_data_split(data_name, mode)
        eval_batches = self.batcher(data=eval_data, bsz=1, shuffle=False)
        return eval_batches

    def load_formatted_preds(self, formatting, data_name, mode):
        probs = self.load_formatted_probs(formatting, data_name, mode)
        preds = {}
        for k, probs in probs.items():
            preds[k] = int(np.argmax(probs, axis=-1))  
        return preds
    
    def load_formatted_probs(self, formatting:str, data_name:str, mode:str):
        """ evaluates the model in the different formatting set up"""
        #if same as default formatting, return default function
        if formatting == self.dir.load_args('model_args.json').formatting:
            return self.load_probs(data_name, mode)
        
        #if predictions not cached, generate and cache them
        if not self.dir.probs_exists(data_name, mode, dir_name=formatting):
            self.set_up_helpers()
            og_formatting = self.data_loader.formatting
            
            self.data_loader.formatting = formatting
            probs = self._probs(data_name, mode)
            self.dir.make_dir(formatting)
            self.dir.save_probs(probs, data_name, mode, dir_name=formatting)
            
            self.data_loader.formatting = og_formatting
        
        #return predictions for this evaluation mode
        probs = self.dir.load_probs(data_name, mode, dir_name=formatting)
        return probs
            
    @staticmethod
    def load_labels(data_name, mode='test'):
        split_index = {'train':0, 'dev':1, 'test':2}
        eval_data = load_data(data_name)[split_index[mode]]
        
        labels_dict = {}
        for k, ex in enumerate(eval_data):
            labels_dict[k] = ex['label']
        return labels_dict

    @staticmethod
    def load_inputs(data_name, mode='test'):
        split_index = {'train':0, 'dev':1, 'test':2}
        eval_data = load_data(data_name)[split_index[mode]]
        
        inputs_dict = {}
        for k, ex in enumerate(eval_data):
            inputs_dict[k] = ex['text']
        return inputs_dict

class EnsembleLoader(SystemLoader):
    def __init__(self, exp_path:str):
        self.exp_path = exp_path
        self.paths  = [f'{exp_path}/{seed}' for seed in os.listdir(exp_path)]
        self.seeds  = [SystemLoader(seed_path) for seed_path in self.paths]
    
    def load_probs(self, data_name, mode)->dict:
        seed_probs = [seed.load_probs(data_name, mode) for seed in self.seeds]

        conv_ids = seed_probs[0].keys()
        assert all([i.keys() == conv_ids for i in seed_probs])

        ensemble = {}
        for conv_id in conv_ids:
            probs = [seed[conv_id] for seed in seed_probs]
            probs = np.mean(probs, axis=0)
            ensemble[conv_id] = probs
        return ensemble    

    def load_formatted_probs(self, formatting, data_name, mode)->dict:
        seed_probs = [seed.load_formatted_probs(formatting, data_name, mode) for seed in self.seeds]
        
        conv_ids = seed_probs[0].keys()
        assert all([i.keys() == conv_ids for i in seed_probs])

        ensemble = {}
        for conv_id in conv_ids:
            probs = [seed[conv_id] for seed in seed_probs]
            probs = np.mean(probs, axis=0)
            ensemble[conv_id] = probs
        return ensemble    
    
    def load_formatted_preds(self, formatting, data_name, mode):
        probs = self.load_formatted_probs(formatting, data_name, mode)
        preds = {}
        for k, probs in probs.items():
            preds[k] = int(np.argmax(probs, axis=-1))
        return preds