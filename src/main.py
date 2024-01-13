import time
import Environment

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--use_gpu", help="Use GPU if True, use CPU if False", action='store_true')
parser.add_argument("--n_epoch", help="Number of epochs to train the machine learning model.", default=200, type=int)
args = parser.parse_args()

t1 = time.time()
env = Environment.Environment(gpu=args.use_gpu)
env.Stock() # Stock supervision data
env.Train(n_epoch=args.n_epoch) # Train machine learning model
t2 = time.time()
print("time: {:.3f} seconds".format(t2-t1))
env.Test_one()

