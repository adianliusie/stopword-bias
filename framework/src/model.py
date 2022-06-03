import torch
import torch.nn as nn

from .utils.torch_utils import load_transformer, load_transformer

class TransformerModel(torch.nn.Module):
    """basic transformer model for multi-class classification""" 
    def __init__(self, trans_name:str, num_classes:int=2):
        super().__init__()
        self.transformer = load_transformer(trans_name)
        h_size = self.transformer.config.hidden_size
        self.output_head = nn.Linear(h_size, num_classes)
        
    def forward(self, *args, **kwargs):
        trans_output = self.transformer(*args, **kwargs)
        H = trans_output.last_hidden_state  #[bsz, L, 768] 
        h = H[:, 0]                         #[bsz, 768] 
        y = self.output_head(h)             #[bsz, C] 
        return y