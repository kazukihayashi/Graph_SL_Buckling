import time
import Environment

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--n_edge_feature", help="Number of edge features to be extracted by graph embedding.", default=100, type=int)
parser.add_argument("--n_whole_feature", help="Number of whole graph features to be extracted by graph embedding", default=100, type=int)
parser.add_argument("--use_gpu", help="Use GPU if enabled, use CPU otherwise", action='store_true')
parser.add_argument("--stock", help="Prepare the samples for training if enabled", action='store_true')
parser.add_argument("--train", help="Implement training if enabled", action='store_true')
parser.add_argument("--n_train_data", help="Number of samples to train the machine learning model.", default=1000, type=int)
parser.add_argument("--n_epoch", help="Number of epochs to train the machine learning model.", default=200, type=int)
args = parser.parse_args()

t1 = time.time()
env = Environment.Environment(gpu=args.use_gpu,n_edge_feature=args.n_edge_feature,n_whole_feature=args.n_whole_feature)
if args.stock:
  env.Stock(n_train_data=args.n_train_data) # Stock supervision data
if args.train:
  env.Train(n_epoch=args.n_epoch) # Train machine learning model
t2 = time.time()
print("time: {:.3f} seconds".format(t2-t1))
env.Test_one()

