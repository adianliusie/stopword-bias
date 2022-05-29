class FeatureExtractor():
    '''
        Defines set of methods for extracting features
    '''

    def num_words(sentences):
        return [len(s.split()) for s in sentences]


class Retention(FeatureExtractor):
    '''
        Funtionality:
            - Extract desired feature per sentence
            - Generate retention plot ordered by feature per fraction of class in pred/label
    '''
    def __init__(self, sentences, ys):
        super().__init__(FeatureExtractor)
        
        self.sentences = sentences
        self.ys = ys