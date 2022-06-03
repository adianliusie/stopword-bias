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
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/plot_retention.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # Load system
    system = SystemLoader(args.MODEL_PATH)

    # Load labels and predictions
    sentences = system.load_sentences(args.DATANAME,  mode='test')
    preds  = system.load_preds(args.DATANAME,  mode='test')
    labels = system.load_labels(args.DATANAME, mode='test')


    # Get retention curves and plot
    RG = RetentionGenerator(sentences, ys)
    feats = RG.get_feat(args.FEAT)
    fracs, pos_class_fracs = RG.retention_plot(feats)

    # Plot
    for class_ind, pos_fracs in pos_class_fracs.items():
        plt.plot(fracs[args.ignore:], pos_fracs[args.ignore:], label=f'class {class_ind}')
        plt.ylabel(f'Class Fraction')
        plt.xlabel(f'Retention Fraction')
        # out_file = f'{args.OUT}/feature_{args.FEAT}_class_{class_ind}.png'
        # plt.clf()
    plt.legend()
    out_file = f'{args.OUT}/data_{args.DATANAME}_feature_{args.FEAT}.png'
    plt.savefig(out_file, bbox_inches='tight')
        

    
