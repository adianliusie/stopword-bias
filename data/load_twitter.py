_DESCRIPTION = """\
Twitter Emotions Dataset
This dataset consists of train (16000), val (2000) and test (2000), where each 
data point is a short tweet. Each tweet is classed into one of 
siz emotions: love, joy, fear, anger, surprise, sadness\
"""

_DOWNLOAD_URL = "https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp?select=train.txt"


class TwitterLoader():

    def __init__(self):

        self.CLASS_TO_IND = {
            'love': 0,
            'joy': 1,
            'fear': 2,
            'anger': 3,
            'surprise': 4,
            'sadness': 5,
        }

    def _read_file(self, filepath):
        tweets = []
        class_labels = []
        emotions = []

        with open(filepath, 'r') as f:
            lines = f.readlines()
        lines = [line.rstrip('\n') for line in lines]

        for line in lines:
            items = line.split(';')
            try:
                label = self.CLASS_TO_IND[items[1]]
                class_labels.append(label)
                tweets.append(items[0])
                emotions.append(items[1])
            except:
                print("Failed to convert class", items[1])
                emotions.append(items[1])
        print("Emotions", list(set(emotions)))
        return tweets, class_labels

    def get_data(self, filepath):
        tweets_list, labels = self._read_file(filepath)
        return tweets_list, labels

    # def get_train(self, filepath='../data/train.txt'):
    #     return self.get_data(filepath)

    # def get_val(self, filepath='../data/val.txt'):
    #     return self.get_data(filepath)

    # def get_test(self, filepath='../data/test.txt'):
    #     return self.get_data(filepath)
