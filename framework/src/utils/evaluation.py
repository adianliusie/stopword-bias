import numpy as np

def get_accuracy(preds:dict, labels:dict):
    perf = np.zeros(2)
    for sample_id, label in labels.items():
        pred = preds[sample_id]
        perf[0] += np.sum(pred == label)
        perf[1] += 1
    acc = perf[0]/perf[1]
    return acc
