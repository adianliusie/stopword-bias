from collections import defaultdict
from utilities.perplexity import perplexity_all

class FeatureExtractor():
    '''
        Defines set of methods for extracting features
    '''
    def __init__(self, sentences):
        self.sentences = sentences

    def num_words(self):
        return [len(s.split()) for s in self.sentences]
    
    def num_chars(self):
        return [len(s) for s in self.sentences]
    
    def perplexity(self):
        return perplexity_all(self.sentences)



class RetentionGenerator(FeatureExtractor):
    '''
        Funtionality:
            - Extract desired feature per sentence
            - Generate retention plot ordered by feature for fraction of class in pred/label
    '''
    def __init__(self, sentences, ys):
        super().__init__(sentences)
        self.ys = ys
    
    def retention_plot(self, features):

        fracs = [(i+1)/len(features) for i,_ in enumerate(features)]

        items = [(f, y) for f,y in zip(features, self.ys)]
        ordered_items = sorted(items, key=lambda x: x[0])
        ordered_ys = [o[1] for o in ordered_items]

        # retention plot per class
        num_classes = len(set(self.ys))
        pos_class_fracs = defaultdict(list) # key refers to class index
        cum_count = defaultdict(int)

        for i, y in enumerate(ordered_ys):
            for c in num_classes:
                if y==c:
                    cum_count[c] += 1
                pos_class_fracs[c].append(cum_count[c]/(i+1))

        return fracs, pos_class_fracs

