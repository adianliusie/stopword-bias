from data.load_twitter import TwitterLoader

def load_data(dataset, filepath=None, type=None):
    if dataset == 'twitter':
        dl =  TwitterLoader()
        return dl.get_data(filepath)