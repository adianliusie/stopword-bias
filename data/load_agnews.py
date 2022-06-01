from datasets import load_dataset

class AGNewsLoader():
    def __init__(self):
        self.dataset = load_dataset('ag_news')
    
    def get_data(self, part='train'):
        texts = self.dataset[part]['text']
        labels = self.dataset[part]['label']

        return texts, labels