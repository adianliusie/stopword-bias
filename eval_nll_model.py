from framework.src.system_loader import SystemLoader
from nll_model import NllModel

import argparse
import os
import sys
from sklearn.metrics import accuracy_score


if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('DATANAME', type=str, help='dataset e.g. imdb')
    commandLineParser.add_argument('MODEL_PATH', type=str, help='path to trained model')
args = commandLineParser.parse_args()

system = SystemLoader(args.MODEL_PATH)
sentences_dict = system.load_inputs(args.DATANAME,  mode='test')
labels_dict = system.load_labels(args.DATANAME, mode=args.mode='test')

sentences = [sentences_dict[i] for i in range(len(sentences_dict))]
labels = [labels_dict[i] for i in range(len(labels_dict))]

nll_system = NllModel(args.DATANAME, 'stop')
preds_nll = nll_system.load_preds(sentences)

acc = accuracy_score(labels, preds_nll)
print("Accuracy", acc)