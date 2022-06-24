import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from types import SimpleNamespace

from .utils.torch_utils import load_transformer, load_transformer
from .utils.glove_utils import get_glove

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
        return SimpleNamespace(h=h, y=y)
    
class GloveAvgModel(torch.nn.Module):
    """ glove word sverage model """
    def __init__(self, trans_name:str, num_classes:int=2):
        super().__init__()
        glove_dict, glove_embeddings = get_glove()
        self.embeddings = nn.Embedding.from_pretrained(glove_embeddings)
        self.output_head = nn.Linear(300, num_classes)
        
    def forward(self, input_ids, attention_mask):
        attention_mask = attention_mask.unsqueeze(-1)
        glove_embeds = self.embeddings(input_ids)
        avg_embeds = torch.sum(glove_embeds*attention_mask, dim=1)/torch.sum(attention_mask, dim=1)
        y = self.output_head(avg_embeds)    #[bsz, C] 
        return SimpleNamespace(h=avg_embeds, y=y)

class GloveBilstmModel(torch.nn.Module):
    """ glove word sverage model """
    def __init__(self, trans_name:str, num_classes:int=2):
        super().__init__()
        glove_dict, glove_embeddings = get_glove()
        self.embeddings = nn.Embedding.from_pretrained(glove_embeddings)
        self.bilstm = nn.LSTM(300, 150, 2, batch_first=True, dropout=0.4, bidirectional=True)
        self.output_head = nn.Linear(300, num_classes)
        
    def forward(self, input_ids, attention_mask):
        glove_embeds = self.embeddings(input_ids)
        seq_lens = torch.sum(attention_mask, dim=-1).tolist()
        embeds = pack_padded_sequence(glove_embeds, seq_lens, batch_first=True, enforce_sorted=False)
        H, _ = self.bilstm(embeds)
        H , _ = pad_packed_sequence(H, batch_first=True)
        attention_mask = attention_mask.unsqueeze(-1)
        avg_embeds = torch.sum(H*attention_mask, dim=1)/torch.sum(attention_mask, dim=1)
        y = self.output_head(avg_embeds)    #[bsz, C] 
        return SimpleNamespace(h=avg_embeds, y=y)

"""
class GloveGruModel(nn.Module):
    def __init__(self, vtrans_name:str, num_classes:int=2):
        super().__init__()
        glove_dict, glove_embeddings = get_glove()

        self.embeddings = nn.Embedding.from_pretrained(glove_embeddings)
        self.gru = nn.GRU(300, 150, num_layers=2, 
                        bidirectional=True, dropout=0.4)
        self.dense = nn.Linear(300, num_classes)
        self.dropout = nn.Dropout(d_rate)
    
    def forward(self, input_ids, attention_mask):
        embedded = self.dropout(self.embeddings(input_ids))
        # embedded: (sentence_length, batch_size, embedding_dim)

        gru_output, hidden = self.gru(embedded)
        # gru_output: (sentence_length, batch_size, hidden_dim * 2)
        # hidden: (num_layers * 2, batch_size, hidden_dim)
        ## ordered: [f_layer_0, b_layer_0, ...f_layer_n, b_layer n]
        
        # concat the final output of forward direction and backward direction
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        # hidden: (batch_size, hidden_dim * 2)

        output = self.dense(hidden)

        return output
"""