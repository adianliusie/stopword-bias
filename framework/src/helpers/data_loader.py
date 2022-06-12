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
            text  = ex['text']
            label = ex['label']
            
            if self.formatting == 'mask_stopwords':
                for word in sorted(self.stop_word_list, key=lambda x: len(x), reverse=True):
                    text = re.sub(r'(?<=\s)'+ word + r'(?=[^a-zA-Z1-9])', r'[MASK]', text)
                    #^ just accept you won't understand this as regex is impossible
                                
            elif self.formatting == 'mask_content':
                text = self.mask_content(text)
                
            elif self.formatting in ['remove_content', 'shuffle_stopwords']:
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
        if self._in_cache(data_name):
            train, dev, test = self._load_proc_data(data_name)
            train, dev, test = train[:lim], dev[:lim], test[:lim]
        else:
            train, dev, test = load_data(data_name, lim)
            train, dev, test = [self.prep_split(split) for split in (train, dev, test)]
            self._save_proc_data(data_name, train, dev, test)
        return train, dev, test
    
    def get_data_split(self, data_name:str, split:str, lim:int=None):
        split_index = {'train':0, 'dev':1, 'test':2}
        data = self.get_data(data_name, lim)[split_index[split]]
        return data
    
    def __call__(self, *args, **kwargs):
        return self.get_data(*args, **kwargs)
    
    def _in_cache(self, data_name:str)->bool:
        pickle_paths = [self._get_pickle_path(data_name, i, self.formatting) \
                                        for i in ['train', 'dev', 'test']]
        return all([os.path.isfile(path) for path in pickle_paths])

    def _save_proc_data(self, data_name:str, train:list, dev:list, test:list):
        split_names = ['train', 'dev', 'test']
        for k, data in enumerate([train, dev, test]):
            split_name = split_names[k]
            pickle_name = self._get_pickle_path(data_name, split_name, self.formatting)
            with open(pickle_name, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_proc_data(self, data_name)->('train', 'dev', 'test'):
        splits = []
        for split_name in ['train', 'dev', 'test']:
            pickle_name = self._get_pickle_path(data_name, split_name, self.formatting)
            with open(pickle_name, 'rb') as handle:
                data = pickle.load(handle)
                splits.append(data)
        return splits
    
    def _get_pickle_path(self, data_name, split_name, formatting):
        return f'data_cache/{data_name}.{split_name}.{self.formatting}.pkl'
    @property
    def stop_word_list(self):
        if not hasattr(self, '_stop_word_list'):  
            self._stop_word_list = stopwords.words('english')
        return self._stop_word_list
    