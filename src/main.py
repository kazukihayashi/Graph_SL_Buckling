import time
import Environment

### User specified parameters ###
USE_GPU = False
N_EPOCH = 200
#################################

## Train ##
t1 = time.time()
env = Environment.Environment(gpu=USE_GPU)
# env.Stock() # Stock supervision data
env.Train(n_epoch=N_EPOCH) # Train machine learning model
t2 = time.time()
print("time: {:.3f} seconds".format(t2-t1))
env.Test_one()

