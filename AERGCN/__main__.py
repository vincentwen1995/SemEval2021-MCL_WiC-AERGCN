import argparse
import random

import numpy as np
import torch

from AERGCN.cli import Interface
from AERGCN.input_pipeline.models import AEGCN_lite, AERGCN, AEGCN, AERGCN_no_R, AERGCN_no_MHA, FullSemantic, FullSemantic_no_MHA, \
    AERGCN_no_syn_MHA, AERGCN_no_MHIA


def main():
    def bool_str(s):
        if s not in ('True', 'False'):
            raise ValueError('Not a valid boolean string.')
        return s == 'True'
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='training',
                        choices=['training', 'development', 'test'], type=str, help='The mode to run: training, development or test.')
    parser.add_argument('--model_dir', default=None, type=str,
                        help='Specify the checkpoint or saved model to load.')
    parser.add_argument('--log_dir', default=None, type=str,
                        help='Specify the logging directory to save the outputs.')
    parser.add_argument('--DEBUG', default=False,
                        type=bool_str, help='Specify DEBUG mode.')
    parser.add_argument('--model_name', default='AERGCN', type=str,
                        help='Specify the model to run: AERGCN, AERGCN_no_R or AERGCN_no_MHA, FullSemantic, FullSemantic_no_MHA, AERGCN_no_syn_MHA, AERGCN_no_MHIA.')
    parser.add_argument('--embed_model_name',
                        default='xlm-roberta-base', type=str, help='Specify the path to the saved Huggingface\'s transformer model or the name of the provided ones.')
    # parser.add_argument('--embed_model_name1',
    #                     default='xlm-roberta-base', type=str, help='Specify the path to the saved Huggingface\'s transformer model or the name of the provided ones. This is used for the first language of the sentence pairs.')
    parser.add_argument('--multi_lingual', default=True, type=bool_str,
                        help='Specify whether the used semantic embedding model supports multiple languages.')
    # parser.add_argument('--embed_model_name2',
    #                     default='xlm-roberta-base', type=str, help='Specify the path to the saved Huggingface\'s transformer model or the name of the provided ones. This is used for the second language of the sentence pairs. If the language model is multi-lingual, only embed_model_name1 needs to be specified.')
    parser.add_argument('--lang_1', default='en', type=str,
                        help='Specify the language of the first sentence in the sentence pair of the dataset.')
    parser.add_argument('--lang_2', default='en', type=str,
                        help='Specify the language of the second sentence in the sentence pair of the dataset.')
    parser.add_argument('--include_pos_tags', default=False, type=bool_str,
                        help='Flag to train the POS tag embeddings for the syntactic branch.')
    parser.add_argument('--regularization', default='basis',
                        choices=[None, 'basis', 'block'], type=str, help='Specify the weight regularization method for R-GCN: None, basis or block.j')
    parser.add_argument('--num_basis', default=16, type=int,
                        help='Specify the number of bases to use for the weight regulization of R-GCN.')
    parser.add_argument('--fine_tuning', default=True, type=bool_str,
                        help='Flag to fine-tune the BERT embeddings.')
    parser.add_argument('--ft_learning_rate',
                        default=1e-5, type=float)
    parser.add_argument('--ft_l2reg', default=0.0, type=float)
    parser.add_argument('--ft_epoch', default=5, type=int,
                        help='Specify the number of epochs to fine-tune the BERT embeddings.')
    parser.add_argument('--num_dependencies', default=20, type=int,
                        help='Specify the number of types of syntactic relations/dependencies to use.')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--l2reg', default=0.001, type=float)
    parser.add_argument('--num_epoch', default=15, type=int)
    parser.add_argument('--repeats', default=1, type=int,
                        help='Number of repeats to train the model.')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=500, type=int)
    parser.add_argument('--embed_dim', default=768, type=int)
    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--save', default=True, type=bool_str,
                        help='Flag to save the checkpoints and trained model.')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default=torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu'), type=str)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--head', default=6, type=int)
    parser.add_argument('--num_rgcn_layer', default=2, type=int)
    parser.add_argument('--score_function', default='scaled_dot_product',
                        choices=['scaled_dot_product', 'mlp'], type=str)
    parser.add_argument('--force_new', default=False, type=bool_str)
    parser.add_argument('--early_stop', default=5, type=int)
    parser.add_argument('--label_smoothing', default=False, type=bool_str)
    parser.add_argument('--ls_eps', default=0.1, type=float,
                        help='Constant for label smoothing.')

    opt = parser.parse_args()

    if opt.multi_lingual is False:
        raise NotImplementedError('Mono-lingual language models are currently not supported.')

    if opt.model_name != 'AERGCN' and opt.model_name != 'AEGCN' and opt.model_name != 'AEGCN_lite':
        raise NotImplementedError('Other variants of the AERGCN model are currently not supported.')

    model_classes = {
        'AERGCN': AERGCN,
        'AEGCN': AEGCN,
        'AEGCN_lite': AEGCN_lite,
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.num_dependencies1 = opt.num_dependencies
    opt.num_dependencies2 = opt.num_dependencies

    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    cli = Interface(opt)
    cli.run()


if __name__ == '__main__':
    main()
