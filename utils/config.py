# -*- coding: utf-8 -*-#
import argparse
import torch

parser = argparse.ArgumentParser(description='Multiple Intent Detection and Slot Filling Joint Model')

# Dataset and Other Parameters
parser.add_argument('--data_dir', '-dd', help='dataset file path', type=str, default='./data/MixATIS_clean')
parser.add_argument('--save_dir', '-sd', type=str, default='./save/MixATIS_best')
parser.add_argument('--load_dir', '-ld', type=str, default=None)
parser.add_argument('--log_dir', '-lod', type=str, default='./log/MixATIS')
parser.add_argument('--log_name', '-ln', type=str, default='log.txt')
parser.add_argument("--random_state", '-rs', help='random seed', type=int, default=72)
parser.add_argument('--gpu', '-g', action='store_true', help='use gpu', required=False, default=True)

# Training parameters.
parser.add_argument('--num_epoch', '-ne', type=int, default=200)
parser.add_argument('--batch_size', '-bs', type=int, default=16)
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-6)
parser.add_argument("--learning_rate", '-lr', type=float, default=0.001)
parser.add_argument('--dropout_rate', '-dr', type=float, default=0.4)
parser.add_argument('--slot_decoder_dropout_rate', '-sddr', type=float, default=0.0)
parser.add_argument('--gcn_dropout_rate', '-gdr', type=float, default=0.4)
parser.add_argument('--threshold', '-thr', type=float, default=0.5)
parser.add_argument('--early_stop', action='store_true', default=False)
parser.add_argument('--patience', '-pa', type=int, default=10)
parser.add_argument('--intent_loss_alpha', '-lalpha', type=float, default=0.8)
parser.add_argument('--slot_loss_alpha', '-salpha', type=float, default=0.2)


# Model parameters.
parser.add_argument('--alpha', '-alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--word_embedding_dim', '-wed', type=int, default=128)
parser.add_argument('--intent_embedding_dim', '-ied', type=int, default=384)
parser.add_argument('--slot_embedding_dim', '-sed', type=int, default=384)
parser.add_argument('--encoder_hidden_dim', '-ehd', type=int, default=256)
parser.add_argument('--self_attention_hidden_dim', '-sahd', type=int, default=1024)
parser.add_argument('--self_attention_output_dim', '-saod', type=int, default=128)
parser.add_argument('--gcn_output_dim', '-god', type=int, default=384)


args = parser.parse_args()
args.gpu = args.gpu and torch.cuda.is_available()
print(str(vars(args)))
