from operator import neg
from webbrowser import get
import numpy as np
from nltk.corpus import stopwords

def get_distributions(train):

    pos_examples = []
    neg_examples = []

    stop_word_list = stopwords.words('english')

    for ex in train:
        if ex['label'] == 1:
            pos_examples.append(ex['text'])
        else:
            neg_examples.append(ex['text'])

    positive_text = ' '.join(pos_examples)
    negative_text = ' '.join(neg_examples)
    positive_words = positive_text.split()
    negative_words = negative_text.split()

    word_distribution = {}
    tot_pos = 0
    for word in positive_words:
        if word in stop_word_list:
            tot_pos += 1
            if word in word_distribution.keys():
                word_distribution[word][0] += 1
            else:
                word_distribution[word] = [1,0]

    tot_neg = 0
    for word in negative_words:
        if word in stop_word_list:
            tot_neg += 1
            if word in word_distribution.keys():
                word_distribution[word][1] += 1
            else:
                word_distribution[word] = [0,1]

    stp_words = stop_word_list
    pos_counts = []
    neg_counts = []
    for w in stp_words:
        if w in word_distribution.keys():
            pos_counts.append(word_distribution[w][0])
            neg_counts.append(word_distribution[w][1])
        else:
            pos_counts.append(0)
            neg_counts.append(0)
    
    return stp_words, np.asarray(pos_counts)/tot_pos, np.asarray(neg_counts)/tot_neg

def unigram_all(sentences, train_data):
    # Construct the distributions for the postive/negative text first
    stp_words, pos_dist, neg_dist = get_distributions(train_data)
    # get nll ratio for each sentence
    all_nll_ratios = []
    for sentence in sentences:
        pos_likelihood = 0
        neg_likelihood = 0
        for w in sentence.split():
            # only perform on stopwords
            if w not in stp_words:
                continue
            idx = stp_words.index(w)
            pos_likelihood += np.log(pos_dist[idx])
            neg_likelihood +=  np.log(neg_dist[idx])
        all_nll_ratios.append(pos_likelihood/neg_likelihood)
    return all_nll_ratios