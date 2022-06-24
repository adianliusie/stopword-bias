import argparse
import os
import shutil
import pprint

from src.trainer import Trainer

#### ArgParse for Model details
model_parser = argparse.ArgumentParser(description='Arguments for system and model configuration')

group = model_parser.add_mutually_exclusive_group(required=True)
group.add_argument('--exp_name', type=str,         help='name to save the experiment as')
group.add_argument('--temp', action='store_true',  help='if set, exp will be saved in temp dir', )

model_parser.add_argument('--transformer',  default='bert',     type=str,  help='[bert, roberta, electra ...]')
model_parser.add_argument('--formatting',   default=None,       type=str,  help='[None, mask_content, shuffle_stopwords]')
model_parser.add_argument('--max_len',      default=512,        type=int,  help='max length of transformer inputs')
model_parser.add_argument('--device',       default='cuda',     type=str,  help='device to use [cuda, cpu]')

model_parser.add_argument('--num_seeds',  default=1,           type=int,  help='number of seeds to train')
model_parser.add_argument('--force',      action='store_true',  help='if set, will overwrite any existing directory')

#### ArgParse for Training details
train_parser = argparse.ArgumentParser(description='Arguments for training the system')

train_parser.add_argument('--data_set',  default='imdb',  type=str,  help='')
train_parser.add_argument('--lim',       default=None,    type=int, help='size of data subset to use (for debugging)')
train_parser.add_argument('--print_len', default=100,     type=int,  help='logging training print size')

train_parser.add_argument('--epochs',  default=4,     type=int,     help='numer of epochs to train')
train_parser.add_argument('--lr',      default=1e-5,  type=float,   help='training learning rate')
train_parser.add_argument('--bsz',     default=8,     type=int,     help='training batch size')

train_parser.add_argument('--optim',   default='adamw', type=str,  help='[adam, adamw, sgd]')
train_parser.add_argument('--wandb',   default=None,    type=str,  help='experiment name to use for wandb (and to enable)')

train_parser.add_argument('--no_save', action='store_false', dest='save', help='whether to not save model')

if __name__ == '__main__':
    model_args = model_parser.parse_known_args()[0]
    train_args = train_parser.parse_known_args()[0]
    
    pprint.pprint(model_args.__dict__)
    print()
    pprint.pprint(train_args.__dict__)
    print()
    
    # Overwrites directory if it exists
    if model_args.force:
        exp_name = model_args.exp_name
        exp_folders = exp_name.split('/')
        if exp_folders[0] == 'trained_models' and os.path.isdir(exp_name) and len(exp_folders)>2:
            shutil.rmtree(exp_name)

    # Train system
    if model_args.num_seeds == 1:
        trainer = Trainer(model_args.exp_name, model_args)
        trainer.bias_train(train_args, 'trained_models/imdb/spurious')
    else:
        for i in range(model_args.num_seeds):
            exp_name = model_args.exp_name + '/' + str(i)
            trainer = Trainer(exp_name, model_args)
            trainer.bias_train(train_args, 'trained_models/imdb/spurious')
