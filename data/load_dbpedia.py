from datasets import load_dataset

class DBpediaLoader():
    def __init__(self):
        self.dataset = load_dataset('dbpedia_14')
    
    def _get_data(self, part='train'):
        texts = self.dataset[part]['content']
        labels = self.dataset[part]['label']

        return texts, labels