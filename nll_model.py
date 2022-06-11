from framework.src.utils.data_utils import load_data
from utilities.unigram import get_distributions
import numpy as np

class NllModel:
    def __init__(self, data_name="imdb", dist_type='stop'):

        train, dev, test = load_data(data_name)
        self.train = train
        self.dev = dev
        self.test = test
        
        # currently only implemented for stop words
        stp_words, pos_dist, neg_dist = get_distributions(self.train)
        self.words = stp_words
        self.pos_dist = pos_dist
        self.neg_dist = neg_dist
        self.thresh = 0

    def load_preds(self, sentences):
        # Assume evaluation on the test split for now
        all_preds = []
        for sentence in sentences:
            pos_likelihood = 0
            neg_likelihood = 0
            for w in sentence.split():
                # only perform on stopwords
                if w not in self.words:
                    continue
                idx = self.words.index(w)
                pos_likelihood += np.log(self.pos_dist[idx])
                neg_likelihood +=  np.log(self.neg_dist[idx])
            nll_ratio = pos_likelihood-neg_likelihood
            if nll_ratio > self.thresh:
                all_preds.append(1)
            else:
                all_preds.append(0)
        return all_preds

    
    