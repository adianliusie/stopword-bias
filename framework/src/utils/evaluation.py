import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def get_accuracy(preds:dict, labels:dict):
    perf = np.zeros(2)
    for sample_id, label in labels.items():
        pred = preds[sample_id]
        perf[0] += np.sum(pred == label)
        perf[1] += 1
    acc = perf[0]/perf[1]
    return acc

def get_loss(probs:dict, labels:dict):
    loss = 0
    for sample_id, label in labels.items():
        prob = probs[sample_id]
        loss -= np.log(prob[label])
    loss /= len(labels)
    return loss

def plot_ROC(probs:dict, labels:dict):
    ex_keys = probs.keys()
    probs  = np.array([probs[k][1] for k in ex_keys]) 
    labels = np.array([labels[k]   for k in ex_keys]) 
    fpr, tpr, thresholds = metrics.roc_curve(labels, probs)
    plt.plot(fpr, tpr, marker='.')

    