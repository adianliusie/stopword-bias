'''
Plot label/prediction (or both) retention plots per class
'''

import argparse
import os
import sys
import matplotlib.pyplot as plt
from data.dataloader import load_data
from feature_extractor import RetentionGenerator

import subprocess
subprocess.run(['export PYTHONPATH=/path/to/parent'])

if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('DATANAME', type=str, help='dataset e.g. twitter')
    commandLineParser.add_argument('FEAT', type=str, help='e.g. num_chars')
    commandLineParser.add_argument('OUT', type=str, help='Directory to save output figures')
    commandLineParser.add_argument('--data_path', type=str, default='none', help='data filepath')
    commandLineParser.add_argument('--data_type', type=str, default='none', help='e.g. train or test')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/plot_retention.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    if args.data_path == 'none':
        data_path = None
    else:
        data_path = args.data_path
    if args.data_type == 'none':
        data_type = None
    else:
        data_type = args.data_type
    
    # Get retention curves and plot
    sentences, ys = load_data(args.DATANAME, data_path, data_type)
    RG = RetentionGenerator(sentences, ys)

    if args.FEAT == 'num_chars':
        feats = RG.num_chars()
    fracs, pos_class_fracs = RG.retention_plot(feats)

    # Plot
    for class_ind, pos_fracs in pos_class_fracs.items():
        plt.plot(fracs, pos_class_fracs)
        plt.ylabel(f'Class {class_ind} Fraction')
        plt.xlabel(f'Retention Fraction')
        out_file = f'{args.OUT}/feature_{args.FEAT}_class_{class_ind}.png'
        plt.savefig(out_file, bbox_inches='tight')
        plt.clf()

    
