import random

from tqdm import tqdm 
from copy import deepcopy
from typing import List, Dict, Tuple
from datasets import load_dataset

def load_data(data_name:str, lim:int=None)->Tuple['train', 'dev', 'test']:
    if data_name == 'imdb':    return _load_imdb(lim)
    if data_name == 'dbpedia': return _load_dbpedia(lim)
    if data_name == 'rt':      return _load_rotten_tomatoes(lim)
    if data_name == 'sst':     return _load_sst(lim)
    if data_name == 'twitter': return _load_twitter(lim)
    if data_name == 'yelp':    return _load_yelp(lim)
    if data_name == 'cola':    return _load_cola(lim)
    if data_name == 'boolq':   return _load_boolq(lim)
    if data_name == 'rte':     return _load_rte(lim)
    else: raise ValueError('invalid dataset provided')

def _load_imdb(lim:int=None)->List[Dict['text', 'label']]:
    dataset = load_dataset("imdb")
    train_data = list(dataset['train'])[:lim]
    train, dev = _create_splits(train_data, 0.8)
    test       = list(dataset['test'])[:lim]
    return train, dev, test

def _load_dbpedia(lim:int=None):
    dataset = load_dataset("dbpedia_14")
    print('loading dbpedia- hang tight')
    train_data = dataset['train'][:lim]
    train_data = [_key_to_text(ex) for ex in tqdm(train_data)]
    train, dev = _create_splits(train_data, 0.8)
        
    test  = dataset['test'][:lim]
    test = [_key_to_text(ex) for ex in test]
    return train, dev, test

def _load_rotten_tomatoes(lim:int=None):
    dataset = load_dataset("rotten_tomatoes")
    train = list(dataset['train'])[:lim]
    dev   = list(dataset['validation'])[:lim]
    test  = list(dataset['test'])[:lim]
    return train, dev, test

def _load_yelp(lim:int=None):
    dataset = load_dataset("yelp_polarity")
    train_data = list(dataset['train'])[:lim]
    train, dev = _create_splits(train_data, 0.8)
    test       = list(dataset['test'])[:lim]
    return train, dev, test
    
def _load_sst(lim:int=None)->List[Dict['text', 'label']]:
    dataset = load_dataset("gpt3mix/sst2")
    train = list(dataset['train'])[:lim]
    dev   = list(dataset['validation'])[:lim]
    test  = list(dataset['test'])[:lim]
    
    train = [_invert_labels(ex) for ex in train]
    dev   = [_invert_labels(ex) for ex in dev]
    test  = [_invert_labels(ex) for ex in test]
    return train, dev, test

def _load_twitter(lim:int=None, balance=True)->List[Dict['text', 'label']]:
    base_path = '/home/alta/BLTSpeaking/exp-vr313/Emotion/data/'
    CLASS_TO_IND = {
        'love': 1,
        'joy': 1,
        'fear': 0,
        'anger': 0,
        'surprise': 1,
        'sadness': 0,
    }
    train = _read_file(f'{base_path}train.txt', CLASS_TO_IND)
    dev = _read_file(f'{base_path}val.txt', CLASS_TO_IND)
    test = _read_file(f'{base_path}test.txt', CLASS_TO_IND)
    if balance:
        "balance test set"
        random.seed(100)
        pos_samples = [t for t in test if t['label']==1]
        neg_samples = [t for t in test if t['label']==0]
        neg_samples = random.sample(neg_samples, len(pos_samples))
        test = pos_samples + neg_samples
    return train, dev, test

def _load_cola(lim:int=None)->List[Dict['text', 'label']]:
    dataset = load_dataset("glue", "cola")
    train = list(dataset['train'])[:lim]
    dev   = list(dataset['validation'])[:lim]

    train = [_key_to_text(ex, old_key='sentence') for ex in train]
    dev   = [_key_to_text(ex, old_key='sentence') for ex in dev]

    return train, dev, dev

def _load_boolq(lim:int=None)->List[Dict['text', 'label']]:
    dataset = load_dataset("super_glue", "boolq")
    train = list(dataset['train'])[:lim]
    dev   = list(dataset['validation'])[:lim]

    train = [_key_to_text(ex, old_key='question') for ex in train]
    dev = [_key_to_text(ex, old_key='question') for ex in dev]

    return train, dev, dev

def _load_rte(lim:int=None)->List[Dict['text', 'label']]:
    dataset = load_dataset("super_glue", "rte")
    train = list(dataset['train'])[:lim]
    dev   = list(dataset['validation'])[:lim]

    train = [_key_to_text(ex, old_key='hypothesis') for ex in train]
    dev = [_key_to_text(ex, old_key='hypothesis') for ex in dev]

    return train, dev, dev


def _read_file(filepath, CLASS_TO_IND):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [line.rstrip('\n') for line in lines]

    examples = []
    for line in lines:
        items = line.split(';')
        try:
            examples.append({'text':items[0], 'label':CLASS_TO_IND[items[1]]})
        except:
            pass
    return examples

def _create_splits(examples:list, ratio=0.8)->Tuple[list, list]:
    examples = deepcopy(examples)
    split_len = int(ratio*len(examples))
    
    random.seed(1)
    random.shuffle(examples)
    
    split_1 = examples[:split_len]
    split_2 = examples[split_len:]
    return split_1, split_2

def _key_to_text(ex:dict, old_key='content'):
    """ convert key name from the old_key to 'text' """
    ex = ex.copy()
    ex['text'] = ex.pop(old_key)
    return ex

def _invert_labels(ex:dict):
    ex = ex.copy()
    ex['label'] = 1 - ex['label']
    return ex

def _map_labels(ex:dict, map_dict={-1:0, 1:1}):
    ex = ex.copy()
    ex['label'] = map_dict[ex['label']]
    return ex
