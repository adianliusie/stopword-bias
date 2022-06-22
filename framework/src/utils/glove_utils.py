import torch
import numpy as np
from types import SimpleNamespace
from collections import defaultdict
import re

def get_glove()->(dict, list):
    vocab_to_id = {}
    embeddings  = []
    
    glove_path = '/home/alta/Conversational/OET/al826/2022-old/data/misc/glove.300d'
    with open(glove_path, 'r') as f:
        for line in f.readlines():
            token, *embed = line.strip().split()
            token = token.lower()

            if token not in vocab_to_id and len(embed) == 300:
                idx = len(vocab_to_id)
                vocab_to_id[token] = idx
                embeddings.append([float(i) for i in embed])
    
    embeddings = torch.FloatTensor(embeddings)
    return vocab_to_id, embeddings

class GloveTokenizer:
    def __init__(self):
        vocab_to_id, _ = get_glove()
        
        #make compatible with unknown tokens
        unk_id = len(vocab_to_id)-1
        self.vocab_to_id = defaultdict(lambda: unk_id, vocab_to_id) #sets UNK to -1

        self.id_to_vocab = {v:k for k, v in self.vocab_to_id.items()}
        self.id_to_vocab[unk_id] = '[UNK]'
        
    def tokenize(self, x):
        words = re.sub('([.,!?()])', r' \1 ', x) #adds space before all punctation
        words = words.split()                    #separates all spaces (^and multi spaces)
        
        input_ids = [self.vocab_to_id[w] for w in words]
        attention_mask = [1]*len(input_ids)
        return SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)
    
    def decode(self, y):
        if isinstance(y, torch.Tensor):
            y = t.tolist()
        
        output = [self.id_to_vocab[i] for i in y]
        return output
            
    def __call__(self, x):
        return self.tokenize(x)
    