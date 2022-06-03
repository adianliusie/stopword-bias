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
positive_words = positive_text.split()
negative_words = negative_text.split()

word_distribution = {}
tot_pos = 0
for word in positive_text:
    if word in stop_word_list:
        tot_pos += 1
        if word in word_distribution.keys():
            word_distribution[word][0] += 1
        else:
            word_distribution[word] = (1,0)

tot_neg = 0
for word in negative_text:
    if word in stop_word_list:
        tot_neg += 1
        if word in word_distribution.keys():
            word_distribution[word][1] += 1
        else:
            word_distribution[word] = (0,1)

stp_words = word_distribution.keys()
pos_counts = []
neg_counts = []
for w in stp_words:
    pos_counts.append(word_distribution[w][0])
    neg_counts.append(word_distribution[word][1])

plt.bar(stp_words, np.asarray(pos_counts)/tot_pos)
plt.xlabel("Stop Words")
plt.ylabel("Fraction")
plt.savefig("pos.png")
plt.clf()

plt.bar(stp_words, np.asarray(neg_counts)/tot_neg)
plt.xlabel("Stop Words")
plt.ylabel("Fraction")
plt.savefig("neg.png")
plt.clf()



