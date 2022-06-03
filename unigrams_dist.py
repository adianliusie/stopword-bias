import numpy as np
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

from framework.src.utils.data_utils import load_data

train, dev, test = load_data('imdb')

# focus just on the train set for now

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

pos_distribution = {}
tot_pos = 0
for word in positive_text:
    if word in stop_word_list:
        tot_pos += 1
        if word in pos_distribution.keys():
            pos_distribution[word] += 1
        else:
            pos_distribution[word] = 1

neg_distribution = {}
tot_neg = 0
for word in negative_text:
    if word in stop_word_list:
        tot_neg += 1
        if word in neg_distribution.keys():
            neg_distribution[word] += 1
        else:
            neg_distribution[word] = 1

plt.bar(pos_distribution.keys(), list(pos_distribution.values())/tot_pos)
plt.xlabel("Stop Words")
plt.ylabel("Fraction")
plt.savefig("pos.png")
plt.clf()

plt.bar(neg_distribution.keys(), list(neg_distribution.values())/tot_neg)
plt.xlabel("Stop Words")
plt.ylabel("Fraction")
plt.savefig("neg.png")
plt.clf()



