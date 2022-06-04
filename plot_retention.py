'''
Plot label/prediction (or both) retention plots per class (cumulative plots)
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
    commandLineParser.add_argument('--class_ind', type=int, default=0, help='target class')
    commandLineParser.add_argument('--mode', type=str, default='test', help='mode of data')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/plot_retention.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # Load system
    system = SystemLoader(args.MODEL_PATH)

    # Load labels and predictions
    sentences_dict = system.load_inputs(args.DATANAME,  mode=args.mode)
    preds_dict  = system.load_preds(args.DATANAME,  mode=args.mode)
    labels_dict = system.load_labels(args.DATANAME, mode=args.mode)

    sentences = [sentences_dict[i] for i in range(len(sentences_dict))]
    preds = [preds_dict[i] for i in range(len(preds_dict))]
    labels = [labels_dict[i] for i in range(len(labels_dict))]

    # Get retention curves
    RG_pred = RetentionGenerator(sentences, preds)
    RG_label = RetentionGenerator(sentences, labels)

    feats_pred = RG_pred.get_feat(args.FEAT)
    fracs, pos_class_fracs_pred = RG_pred.retention_plot(feats_pred, cum=True)
    feats_label = RG_label.get_feat(args.FEAT)
    _, pos_class_fracs_label = RG_label.retention_plot(feats_label, cum=True)

    # Plot
    plt.plot(fracs, pos_class_fracs_pred[args.class_ind], label=f'pred class {args.class_ind}')
    plt.plot(fracs, pos_class_fracs_label[args.class_ind], label=f'label class {args.class_ind}')
    plt.ylabel(f'Cumulative Class Fraction')
    plt.xlabel(f'Retention Fraction')
    plt.legend()

    # Save
    model_name = args.MODEL_PATH
    model_name = model_name.split('/')
    model_name = model_name[-2:]
    model_name = '-'.join(model_name)
    out_file = f'{args.OUT}/data_{args.DATANAME}_feature_{args.FEAT}_mode_{args.mode}_model_{model_name}.png'
    plt.savefig(out_file, bbox_inches='tight')
        

    
