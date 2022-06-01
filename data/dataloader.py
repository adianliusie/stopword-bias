from data.load_twitter import TwitterLoader
from data.load_imdb import IMDBLoader
from data.load_agnews import AGNewsLoader
from data.load_dbpedia import DBpediaLoader

def load_data(dataset, filepath=None, type=None):
    if dataset == 'twitter':
        dl =  TwitterLoader()
        return dl.get_data(filepath)
    if dataset == 'imdb':
        dl = IMDBLoader()
        return dl.get_data(filepath)
    if dataset == 'agnews':
        dl = AGNewsLoader(part=type)
    if dataset == 'dbpedia':
        dl = DBpediaLoader(part=type)