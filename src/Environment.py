import numpy as np
np.random.seed(1000)
import csv
import Plotter
import Agent
import pickle

### User specified parameters ###
import TrussEnv as env
RECORD_INTERVAL = 10
#################################

class Environment():
	def __init__(self,gpu,n_edge_feature,n_whole_feature):
		self.env = env.Truss()
		self.env.reset(test=True)
		self.brain = Agent.Brain(self.env.nfv,self.env.nfw,n_edge_feature,n_whole_feature,gpu)
		if gpu:
			self.brain.model = self.brain.model.to("cuda")
		pass

	def Stock(self,n_train_data):
		max_nx = 6
		max_ny = 6
		min_nx = 5
		min_ny = 5
		if max_nx < min_nx:
			assert "Invalid max_nx"
		if max_ny < min_ny:
			assert "Invalid max_ny"
		count = 0
		for i in range(n_train_data):
			self.env.reset(nx=np.randint(min_nx,max_nx),ny=np.randint(min_ny,max_ny))
			v,w,target = self.env.run()
			if target != None:
				self.brain.store(self.env.connectivity,v,w,target)
				count += 1
			if count%100==0:
				print("Stocked {0} data.".format(count))
		print("Finish. Stocked {0} data.".format(count))
		with open('memory.pickle','wb') as f:
			pickle.dump(self.brain.memory, f)
		

	def Train(self,n_epoch):        
		with open('memory.pickle', 'rb') as f:
			self.brain.memory = pickle.load(f)

			targets = np.array([self.brain.memory[i][3] for i in range(len(self.brain.memory))])
			# import matplotlib.pyplot as plt
			# plt.rcParams["font.family"] = "Times New Roman"
			# plt.rcParams["font.size"] = 30
			# plt.rcParams["mathtext.fontset"] = 'stix' # 'stix' 'cm'
			# plt.rcParams['mathtext.default'] = 'it'
			# fig = plt.figure(figsize=(10,4),frameon=False,constrained_layout=True)
			# plt.xlabel("data",labelpad=5)
			# plt.ylabel(r"$\alpha_\mathrm{true}$",rotation=90,labelpad=5)
			# plt.bar(np.arange(len(targets)), np.sort(targets), width=1.0,color="#777777")
			# plt.xlim(1,40000)
			# plt.ylim(0.39,0.76)
			# plt.xticks(np.arange(0,40000+1,10000))
			# plt.yticks(np.arange(0.4,0.75,0.1))
			# plt.show()
			print("Max. target value: {0}".format(targets.max()))
			print("Min. target value: {0}".format(targets.min()))
			print("Ave. target value: {0}".format(targets.mean()))

		self.brain.train(n_epoch=n_epoch)

	# def Test(self):
	#     self.env.reset(test=True)
	#     self.brain = agent.Brain(v.shape[1],w.shape[1],N_EDGE_FEATURE,False)
	#     self.brain.model.Load(filename="trained_model")
	#     with open('memory.pickle', 'rb') as f:
	#         self.brain.memory = pickle.load(f)
	#     self.brain.validate(gpu=False)

	def Test_one(self):
		self.env.reset(nx=4,ny=4,test=False)
		self.brain = Agent.Brain(self.env.nfv,self.env.nfw,N_EDGE_FEATURE,N_WHOLE_FEATURE,False)
		self.brain.model.Load(filename="trained_model")
		
		# with open('memory.pickle', 'rb') as f:
		# 	self.brain.memory = pickle.load(f)
		# 	targets = np.array([self.brain.memory[i][3] for i in range(len(self.brain.memory))])
		# 	imin = np.argmin(targets)
		# 	imax = np.argmax(targets)

		v,w,target = self.env.run()
		prediction = self.brain.model.Forward(v,w,self.env.connectivity).cpu().detach().numpy()
		self.env.render(name="init")
		print("prediction={}".format(prediction))
		print("true      ={}".format(target))
		return prediction, target
