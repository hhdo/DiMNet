import argparse
from utils import get_rank
import wandb

parser = argparse.ArgumentParser(description='DiMNet')

parser.add_argument("--gpus", nargs='+', type=int, default=[0],
                    help="gpus")
parser.add_argument("--batch_size", type=int, default=1,
                    help="batch size")
parser.add_argument("--n_worker", type=int, default=0,
                    help="number of workers for dataloader")
parser.add_argument("-d", "--dataset", type=str, default='ICEWS18',
                    help="dataset to use")
parser.add_argument("--test", action='store_true', default=False,
                    help="load stat from dir and directly test")
parser.add_argument("--pretrain_name", type=str, default=None,
                    help="specify the pretrain_name if this is TEST mode")
parser.add_argument("-C", "--comment", type=str, default='default',
                    help="Comments for logging")

# configuration for stat training
parser.add_argument("--n_epoch", type=int, default=60,
                    help="number of minimum training epochs on each time step")
parser.add_argument("--lr", type=float, default=1e-3,
                    help="learning rate")
parser.add_argument("--decay", type=float, default=1e-4,
                    help="weight decay")
parser.add_argument("--grad_norm", type=float, default=1.0,
                    help="norm to clip gradient to")
parser.add_argument("--patience", type=int, default=10,
                    help="patience for early stopping")
         

# configuration for evaluating
parser.add_argument("--metric", type=list, default=['mrr', 'hits@1', 'hits@3', 'hits@10'],
                    help="evaluating metrics")


# configuration for sequences stat
parser.add_argument("--history_len", type=int, default=10,
                    help="history length")
parser.add_argument("--topk", type=int, default=50,
                    help="generate topk edges for virtual graph")


# configuration for layers
parser.add_argument("--input_dim", type=int, default=128,
                    help="dimension of layer input")
parser.add_argument("--num_head", type=int, default=1,
                    help="number of heads for multi-head attention")
parser.add_argument("--message_func", type=str, default='transe',
                    help="which message_func you use")
parser.add_argument("--aggregate_func", type=str, default='pna',
                    help="which aggregate_func you use")
parser.add_argument("--num_ly", type=int, default=3,
                    help="number of layers")

parser.add_argument("--short_cut", action='store_true', default=True,
                    help="whether residual connection")
parser.add_argument("--layer_norm", action='store_true', default=True,
                    help="whether layer_norm")   

# configuration for decoder
parser.add_argument("--input_dropout", type=float, default=0.2,
                    help="input dropout for decoder ")
parser.add_argument("--hidden_dropout", type=float, default=0.2,
                    help="hidden dropout for decoder")
parser.add_argument("--feat_dropout", type=float, default=0.2,
                    help="feat dropout for decoder")   

args, unparsed = parser.parse_known_args()

wandb.init(project="DiMNet", name=args.comment)

if len(wandb.config.items()) != 0:
  for key, value in wandb.config.items():
    args.__dict__[key] = value

wandb.config.update(vars(args))

if get_rank() == 0:
  print(args)  
  print(unparsed)
  
