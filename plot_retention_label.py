'''
    Plot retention plot for labels of multiple datasets
'''

import argparse
import os
import sys
import matplotlib.pyplot as plt
from framework.src.system_loader import EnsembleLoader
from feature_extractor import RetentionGenerator

if __name__ == '__main__':

    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--datanames',
                        help='dataset e.g. imdb twitter.', nargs='+'
                        required=True)
    parser.add_argument('--feat',
                        help='e.g. unigram.',
                        required=True)
    parser.add_argument('--out',
                        help='Directory to save output figures.',
                        required=True)
    parser.add_argument('--class_ind',
                        type=int,
                        help='target class',
                        default='1')   
    parser.add_argument('--mode',
                        help='mode of data',
                        default='test')
    parser.add_argument('--cum',
                        help='cumulative plot?',
                        default='yes')                      
    args = parser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/plot_retention_label.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    cum = True
    if args.cum == 'no':
        cum = False
    
    for dataname in args.datanames:

        sentences_dict = EnsembleLoader.load_inputs(dataname,  mode=args.mode)
        sentences = [sentences_dict[i] for i in range(len(sentences_dict))]

        labels_dict = EnsembleLoader.load_labels(dataname, mode=args.mode)
        labels = [labels_dict[i] for i in range(len(labels_dict))]

        RG = RetentionGenerator(sentences, labels)

        feats_label = RG.get_feat(args.feat)
        fracs, pos_class_fracs_label = RG.retention_plot(feats_label, cum)
        _, pos_class_fracs_ideal = RG.retention_plot(labels, cum)

        plt.plot(fracs, pos_class_fracs_label[args.class_ind], label=f'{dataname}')

    plt.plot(fracs, fracs, label=f'no bias', linestyle='dashed')
    plt.plot(fracs, pos_class_fracs_ideal[args.class_ind], label=f'full bias', linestyle='dashed')
    plt.ylabel(f'Total Positive Class Fraction')
    plt.xlabel(f'Retention Fraction')
    plt.legend()

    out_file = f'{args.out}/all_data_feature_{args.feat}_mode_{args.mode}.png'
    plt.savefig(out_file, bbox_inches='tight')