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

plt.barh(stp_words, np.asarray(pos_counts)/tot_pos)
plt.barh(stp_words, np.asarray(neg_counts)/tot_neg)
plt.ylabel("Stop Words")
plt.xlabel("Fraction")
plt.xlim([0,0.13])
plt.savefig("pos_neg.png")
# plt.clf()


# plt.xlabel("Stop Words")
# plt.ylabel("Fraction")

# plt.savefig("neg.png")
# plt.clf()



