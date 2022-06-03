import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from .trainer import Trainer
from .utils.torch_utils import no_grad
from .utils.data_utils import load_data
from .helpers import DirHelper

class SystemLoader(Trainer):
    """Base loader class- the inherited class inherits
       the Trainerso has all experiment methods"""

    def __init__(self, exp_path:str):
        self.dir = DirHelper.load_dir(exp_path)
        
    def set_up_helpers(self, args):
        #load training arguments and set up helpers
        super().set_up_helpers(args)

        #load final model
        self.load_model()
        self.model.eval()

    def load_preds(self, data_name, mode):
        probs = self.load_probs(data_name, mode)
        
        preds = {}
        for k, probs in probs.items():
            preds[k] = int(np.argmax(probs, axis=-1))
        
        return preds
        
    def load_probs(self, data_name, mode):
        """loads predictions if saved, else generates"""
        if not self.dir.probs_exists(data_name, mode):
            args = self.dir.load_args('model_args.json')
            self.set_up_helpers(args)
            self.to('cuda:0')
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
        split_index = {'train':0, 'dev':1, 'test':2}
        eval_data = self.data_loader(data_name)[split_index[mode]]
        eval_batches = self.batcher(data=eval_data, bsz=1, shuffle=False)
        return eval_batches

    def load_labels(self, data_name, mode='test'):
        split_index = {'train':0, 'dev':1, 'test':2}
        eval_data = load_data(data_name)[split_index[mode]]
        
        labels_dict = {}
        for k, ex in enumerate(eval_data):
            labels_dict[k] = ex['label']
        return labels_dict
