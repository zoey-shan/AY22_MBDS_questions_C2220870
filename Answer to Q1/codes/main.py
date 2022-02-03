import argparse

from util.model import Model
from util import train, test

parser = argparse.ArgumentParser(description='')

parser.add_argument('--num_instances', default=100, type=int, help='number of instances (patches) in a bag',
                    dest='num_instances')
parser.add_argument('--num_features', default=128, type=int, help='number of features', dest='num_features')
parser.add_argument('--num_bins', default=21, type=int, help='number of bins in distribution pooling filter',
                    dest='num_bins')
parser.add_argument('--sigma', default=0.05, type=float, help='sigma in distribution pooling filter', dest='sigma')
parser.add_argument('--num_classes', default=1, type=int, help='number of classes', dest='num_classes')
parser.add_argument('--batch_size', default=100, type=int, help='batch size', dest='batch_size')
parser.add_argument('--num_epochs', default=100, type=int, help='number of steps of execution (default: 100)',
                    dest='num_epochs')
## batch_size 500

args = parser.parse_args()


def main(batch_size, num_classes, num_instances, num_features, num_bins, num_epochs, sigma):
    model = Model(num_classes=num_classes, num_instances=num_instances, num_features=num_features, num_bins=num_bins,
                  sigma=sigma)
    state_filename = train(model, batch_size=batch_size, num_epochs=num_epochs)
    test(model, state_filename, batch_size=batch_size)


if __name__ == '__main__':
    main(**vars(args))
