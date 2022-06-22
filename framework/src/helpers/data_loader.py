import random 
import pickle 
import os

from types import SimpleNamespace
from tqdm import tqdm
from nltk.corpus import stopwords
import re


from ..utils.data_utils  import load_data
from ..utils.torch_utils import load_tokenizer

class DataLoader:
    def __init__(self, trans_name:str, formatting:str=None):
        self.tokenizer = load_tokenizer(trans_name)
        self.formatting = formatting

    def prep_split(self, data:list):
        random.seed(1) #set random seed for reproducibility
        
        output = []
        for ex in tqdm(data):
            text  = ex['text'].lower()
            label = ex['label']
            
            if self.formatting == 'mask_stopwords':
                for word in sorted(self.stop_word_list, key=lambda x: len(x), reverse=True):
                    text = re.sub(r'(?<=\s)'+ word + r'(?=[^a-zA-Z1-9])', r'[MASK]', text)
                    #^ just accept you won't understand this as regex is impossible
                                
            elif self.formatting == 'mask_content':
                text = self.mask_content(text)
                
            elif self.formatting in ['remove_content', 'shuffle_stopwords']:
                text = re.sub(r'[^\w\s\']',' ', text) #remove all punct except for '
                text = self.mask_content(text)
                text = text.replace('[MASK]', '')
                text = re.sub(' +', ' ', text) #rm multi spaces
                
                if self.formatting == 'shuffle_stopwords':
                    word_list = text.split().copy()
                    random.shuffle(word_list)
                    text = ' '.join(word_list)
            
            elif self.formatting == 'shuffled':
                word_list = text.split().copy()
                random.shuffle(word_list)
                text = ' '.join(word_list)
                
            else:
                assert self.formatting == None
                
            ids = self.tokenizer(text).input_ids
            output.append(SimpleNamespace(text=text, ids=ids, label=label))
        return output
    
    def mask_content(self, text):
        new_text = []
        for word in text.split():
            if word in self.stop_word_list: new_text.append(word)
            else:                           new_text.append('[MASK]')
        new_text = ' '.join(new_text)
        return new_text
                                                                          
    def get_data(self, data_name:str, lim:int=None):
        train, dev, test = load_data(data_name, lim)
        train, dev, test = [self.prep_split(split) for split in (train, dev, test)]
        return train, dev, test
    
    def get_data_split(self, data_name:str, split:str, lim:int=None):
        split_index = {'train':0, 'dev':1, 'test':2}
        data = load_data(data_name)[split_index[split]]
        data = self.prep_split(data)
        return data
    
    def __call__(self, *args, **kwargs):
        return self.get_data(*args, **kwargs)
    
    @property
    def stop_word_list(self):
        if not hasattr(self, '_stop_word_list'):  
            self._stop_word_list = stopwords.words('english')
            self._stop_word_list = [i.lower() for i in self._stop_word_list]
        return self._stop_word_list
    