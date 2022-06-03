from types import SimpleNamespace
from tqdm import tqdm
from nltk.corpus import stopwords

from ..utils.data_utils  import load_data
from ..utils.torch_utils import load_tokenizer

class DataLoader:
    def __init__(self, trans_name:str, formatting:str=None):
        self.tokenizer = load_tokenizer(trans_name)
        self.formatting = formatting
        
        if formatting:
            self.stop_word_list = stopwords.words('english')

    def prep_split(self, data:list):
        output = []
        for ex in tqdm(data):
            text  = ex['text']
            label = ex['label']
            
            if self.formatting == 'mask_stopwords':
                for word in sorted(self.stop_word_list, key=lambda x: len(x), reverse=True):
                    text.replace(word, '[MASK]')
            
            elif self.formatting == 'mask_content':
                new_text = []
                for word in text.split():
                    if word in self.stop_word_list: new_text.append(word)
                    else:                           new_text.append('[MASK]')
                text = ' '.join(new_text)
            else:
                assert self.formatting == None
                
            ids   = self.tokenizer(text).input_ids
            output.append(SimpleNamespace(text=text, ids=ids, label=label))
        return output
    
    def get_data(self, data_name:str, lim:int=None):
        print('tokenizing data')
        train, dev, test = load_data(data_name, lim)
        train = self.prep_split(train)
        dev   = self.prep_split(dev)
        test  = self.prep_split(test)
        return train, dev, test
    
    def __call__(self, *args, **kwargs):
        return self.get_data(*args, **kwargs)