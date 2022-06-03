'''
Plot label/prediction (or both) retention plots per class
'''

import argparse
import os
import sys
import matplotlib.pyplot as plt
from framework.src.system_loader import SystemLoader
from feature_extractor import RetentionGenerator



if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('DATANAME', type=str, help='dataset e.g. imdb')
    commandLineParser.add_argument('MODEL_PATH', type=str, help='path to trained model')
    commandLineParser.add_argument('FEAT', type=str, help='e.g. num_chars')
    commandLineParser.add_argument('OUT', type=str, help='Directory to save output figures')
    commandLineParser.add_argument('--ignore', type=int, default=0, help='num initial points to not plot')
    commandLineParser.add_argument('--class_ind', type=int, default=0, help='target class')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/plot_retention.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # Load system
    system = SystemLoader(args.MODEL_PATH)

    # Load labels and predictions
    sentences_dict = system.load_inputs(args.DATANAME,  mode='test')
    preds_dict  = system.load_preds(args.DATANAME,  mode='test')
    labels_dict = system.load_labels(args.DATANAME, mode='test')

    sentences = [sentences_dict[i] for i in range(len(sentences_dict))]
    preds = [preds_dict[i] for i in range(len(preds_dict))]
    labels = [labels_dict[i] for i in range(len(labels_dict))]

    # Get retention curves
    RG_pred = RetentionGenerator(sentences, preds)
    RG_label = RetentionGenerator(sentences, labels)

    feats_pred = RG_pred.get_feat(args.FEAT)
    fracs, pos_class_fracs_pred = RG_pred.retention_plot(feats_pred)
    feats_label = RG_label.get_feat(args.FEAT)
    _, pos_class_fracs_label = RG_label.retention_plot(feats_label)

    # Plot WORK IN PROGRESS
    for class_ind, pos_fracs in pos_class_fracs.items():
    plt.plot(fracs[args.ignore:], pos_fracs[args.ignore:], label=f'class {args.class_ind}')
    plt.ylabel(f'Class Fraction')
    plt.xlabel(f'Retention Fraction')
    plt.legend()
    out_file = f'{args.OUT}/data_{args.DATANAME}_feature_{args.FEAT}.png'
    plt.savefig(out_file, bbox_inches='tight')
        

    
