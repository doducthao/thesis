import argparse
from email.policy import default

def create_parser():
    parser = argparse.ArgumentParser(description='RMCOS-SSL')
    parser.add_argument('--dataset', default='cifar10'),
    parser.add_argument('--train-subdir', type=str, default='train+val')
    parser.add_argument('--eval-subdir', type=str, default='test')
    parser.add_argument('--labels', default="data-local/labels/cifar10/1000_balanced_labels/00.txt", type=str)
    parser.add_argument('--exclude-unlabeled', type=str2bool, default=False)
    parser.add_argument('--arch', '-a', default='cifar_shakeshake26')

    parser.add_argument('--lrG', type=float, default=0.0002)  
    parser.add_argument('--lrD', type=float, default=0.0002) 
    parser.add_argument('--beta1', type=float, default=0.5)  
    parser.add_argument('--beta2', type=float, default=0.999)

    parser.add_argument('--workers', default=6, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--labeled-batch-size', default=32, type=int)
    parser.add_argument('--generated-batch-size', default=100, type=int)

    parser.add_argument('--z-dim', type=int, default=100)
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float)
    parser.add_argument('--initial-lr', default=0.0, type=float)

    parser.add_argument('--lr-rampup', default=0, type=int)
    parser.add_argument('--lr-rampdown-epochs', default=None, type=int)

    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--nesterov', type=str2bool, default=True)

    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--ema-decay', default=0.999, type=float)

    parser.add_argument('--consistency', default=100.0, type=float)
    parser.add_argument('--consistency-type', default="mse", type=str)
    parser.add_argument('--consistency-rampup', default=5, type=int)
    parser.add_argument('--logit-distance-cost', default=0.01, type=float)

    parser.add_argument('--checkpoint-epochs', default=1, type=int)
    parser.add_argument('--evaluation-epochs', default=1, type=int)

    parser.add_argument('--print-freq', default=20, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--evaluate', action="store_false")
    parser.add_argument('--pretrained', type=str2bool, default=False)
    return parser.parse_args()

# if use str2bool
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# if use str2epoch
def str2epochs(v):
    try:
        if len(v) == 0:
            epochs = []
        else:
            epochs = [int(string) for string in v.split(",")]
    except:
        raise argparse.ArgumentTypeError(
            'Expected comma-separated list of integers, got "{}"'.format(v))
    if not all(0 < epoch1 < epoch2 for epoch1, epoch2 in zip(epochs[:-1], epochs[1:])):
        raise argparse.ArgumentTypeError(
            'Expected the epochs to be listed in increasing order')
    return epochs