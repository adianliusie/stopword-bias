'''
Plot label/prediction (or both) retention plots per class (cumulative plots)
'''

import argparse
import os
import sys
import matplotlib.pyplot as plt
from framework.src.system_loader import EnsembleLoader
from feature_extractor import RetentionGenerator
from nll_model import NllModel



if __name__ == '__main__':

    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname',
                        help='dataset e.g. imdb.',
                        required=True)
    parser.add_argument('--model_paths',
                        help='Path to the model files, arbitrary path for stopgram.', nargs='+',
                        required=True)
    parser.add_argument('--model_names',
                        help='names for models in legend.', nargs='+',
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
    parser.add_argument('--unigram_train_data_name',
                        type=str,
                        help='training data to calculate unigram function',
                        default='imdb') 
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
    with open('CMDs/plot_retention.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # Load system
    systems = []
    for pth, mdl_name in zip(args.model_paths, args.model_names):
        system = EnsembleLoader(pth)

        # Load labels and predictions
        sentences_dict = system.load_inputs(args.dataname,  mode=args.mode)
        sentences = [sentences_dict[i] for i in range(len(sentences_dict))]

        labels_dict = system.load_labels(args.dataname, mode=args.mode)
        labels = [labels_dict[i] for i in range(len(labels_dict))]

        if mdl_name == 'stopgram':
            stp = NllModel()
            preds = stp.load_preds(sentences)
        else:
            preds_dict  = system.load_preds(args.dataname,  mode=args.mode)
            preds = [preds_dict[i] for i in range(len(preds_dict))]
        
        cum = True
        if args.cum == 'no':
            cum = False

        # Get retention curves
        RG_pred = RetentionGenerator(sentences, preds)
        RG_label = RetentionGenerator(sentences, labels)

        feats_pred = RG_pred.get_feat(args.feat, data_name=args.unigram_train_data_name)
        fracs, pos_class_fracs_pred = RG_pred.retention_plot(feats_pred, cum)
        feats_label = RG_label.get_feat(args.feat, data_name=args.unigram_train_data_name)
        _, pos_class_fracs_label = RG_label.retention_plot(feats_label, cum)
        _, pos_class_fracs_ideal = RG_label.retention_plot(labels, cum)


        # Plot
        plt.plot(fracs, pos_class_fracs_pred[args.class_ind], label=f'{mdl_name}')

    plt.plot(fracs, pos_class_fracs_label[args.class_ind], label=f'label')
    plt.plot(fracs, fracs, label=f'no bias', linestyle='dashed')
    plt.plot(fracs, pos_class_fracs_ideal[args.class_ind], label=f'full bias', linestyle='dashed')
    if cum:
        # plt.ylabel(f'Cumulative Class {args.class_ind} Fraction')
        plt.ylabel(f'Total Positive Class Fraction')
    else:
        plt.ylabel(f'Class Fraction')
    plt.xlabel(f'Retention Fraction')
    plt.legend()

    out_file = f'{args.out}/data_{args.dataname}_feature_{args.feat}_mode_{args.mode}.png'
    plt.savefig(out_file, bbox_inches='tight')
        

    
