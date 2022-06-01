from data.load_twitter import TwitterLoader
from data.load_imdb import IMDBLoader

def load_data(dataset, filepath=None, type=None):
    if dataset == 'twitter':
        dl =  TwitterLoader()
        return dl.get_data(filepath)
    if dataset == 'imdb':
        dl = IMDBLoader()
        return dl.get_data(filepath)