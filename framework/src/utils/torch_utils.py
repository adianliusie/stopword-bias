import torch
from typing import Callable
from transformers import BertModel, ElectraModel, RobertaModel
from transformers import BertTokenizerFast, ElectraTokenizerFast, RobertaTokenizerFast

def load_tokenizer(system:str)->'Tokenizer':
    """ downloads and returns the relevant pretrained tokenizer from huggingface """
    if system   == 'bert'          : tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    elif system == 'bert_cased'    : tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    elif system == 'bert_large'    : tokenizer = BertTokenizerFast.from_pretrained("bert-large-uncased")
    elif system == 'roberta'       : tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    elif system == 'electra'       : tokenizer = ElectraTokenizerFast.from_pretrained("bert-base-uncased")
    elif system == 'electra_large' : tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-large-discriminator")
    else: raise ValueError("invalid transfomer system provided")
    return tokenizer

def load_transformer(system:str)->'Model':
    """ downloads and returns the relevant pretrained transformer from huggingface """
    if   system == 'bert'       : trans_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
    elif system == 'bert_cased' : trans_model = BertModel.from_pretrained('bert-base-cased', return_dict=True)
    elif system == 'bert_large' : trans_model = BertModel.from_pretrained('bert-large-uncased', return_dict=True)
    elif system == 'roberta'    : trans_model = RobertaModel.from_pretrained('roberta-base', return_dict=True)
    elif system == 'electra'    : trans_model = ElectraModel.from_pretrained('google/electra-base-discriminator',return_dict=True)
    elif system == 'electra_large':
        trans_model= ElectraModel.from_pretrained('google/electra-large-discriminator', return_dict=True)
    else: raise ValueError("invalid transfomer system provided")
    return trans_model

def no_grad(func:Callable)->Callable:
    """ decorator which detaches gradients """
    def inner(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return inner
