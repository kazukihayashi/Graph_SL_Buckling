import os
import numpy as np
np.random.seed(0)
import random
random.seed(0)
from copy import deepcopy
import torch
torch.manual_seed(0)
from collections import deque
import csv

### User specified parameters ###
INIT_MEAN = 0.0 ## mean of initial training parameters
INIT_STD = 0.005 ## standard deviation of initial training parameters
BATCH_SIZE = 32
USE_BIAS = False
#################################

class GlobalAttentionNetwork(torch.nn.Module):
    def __init__(self,n_node_feature,n_global_feature,use_gpu,use_tanh=False):
        super(GlobalAttentionNetwork,self).__init__()

        self.n_global_feature = n_global_feature
        if use_tanh:
            self.actF = torch.nn.Tanh()
        else:
            self.actF = torch.nn.Identity()
        self.gate_nn = torch.nn.Linear(n_node_feature,1)
        self.nn = torch.nn.Linear(n_node_feature,n_global_feature)
        self.Initialize_weight()

        if use_gpu:
            self.to('cuda')
            self.device = torch.device('cuda')
        else:
            self.to('cpu')
            self.device = torch.device('cpu')

    def Initialize_weight(self):
        for m in self._modules.values():
            if isinstance(m,torch.nn.Linear):
                torch.nn.init.normal_(m.weight,mean=0,std=INIT_STD)

    def Forward(self,x,n_graph,index):
        '''
        x[n_node,n_feature]
        '''
        if n_graph == 1:
            # gate = torch.sigmoid(gate_nn.forward(x)) # original paper
            gate = torch.softmax(self.gate_nn.forward(x),0) # Pytorch Geometric implementation (this is better because the magnitude of output does not depend on the number of nodes)
            x = self.actF(self.nn.forward(x))
            y = torch.sum(gate*x,dim=0)
        else:
            gate_numerator = torch.exp(self.gate_nn.forward(x))
            gate_denominator = torch.zeros(n_graph,1,dtype=torch.float32,device=self.device).scatter_add(dim=0,index=index[:,0:1],src=gate_numerator).index_select(dim=0,index=index[:,0])
            gate = gate_numerator/gate_denominator
            x = self.actF(self.nn.forward(x))
            y = torch.zeros(n_graph,x.shape[1],dtype=torch.float32,device=self.device)
            y = y.scatter_add(dim=0,index=index,src=gate*x)
        return self.actF(y)

class NN(torch.nn.Module):
    def __init__(self,n_node_input,n_edge_input,n_node_output,n_whole_output,use_gpu):
        super(NN,self).__init__()
        self.l1_1 = torch.nn.Linear(n_node_input,n_node_output,bias=USE_BIAS)
        self.l1_2 = torch.nn.Linear(n_edge_input,n_node_output,bias=USE_BIAS)
        self.l1_3 = torch.nn.Linear(n_node_output,n_node_output,bias=USE_BIAS)
        self.l1_4 = torch.nn.Linear(n_node_output,n_node_output,bias=USE_BIAS)
        self.l1_5 = torch.nn.Linear(n_whole_output,n_node_output,bias=USE_BIAS)
        self.l1_6 = torch.nn.Linear(n_node_output*3,n_node_output,bias=USE_BIAS)

        self.l2_1 = torch.nn.Linear(n_whole_output,n_whole_output,bias=USE_BIAS)
        self.l2_2 = torch.nn.Linear(n_whole_output,1,bias=USE_BIAS)

        self.ActivationF = torch.nn.LeakyReLU(0.2)

        self.Initialize_weight()

        self.n_node_output = n_node_output
        self.n_whole_output = n_whole_output
        if use_gpu:
            self.to('cuda')
            self.device = torch.device('cuda')
        else:
            self.to('cpu')
            self.device = torch.device('cpu')

        self.global_attn_NN = GlobalAttentionNetwork(n_node_output,n_whole_output,use_gpu)

    def Initialize_weight(self):
        for m in self._modules.values():
            if isinstance(m,torch.nn.Linear):
                torch.nn.init.normal_(m.weight,mean=0,std=INIT_STD)

    def Connectivity(self,connectivity,n_nodes):
        '''
        connectivity[n_edges,2]
        '''
        n_edges = connectivity.shape[0]
        order = np.arange(n_edges)
        adjacency = torch.zeros(n_nodes,n_nodes,dtype=torch.float32,device=self.device,requires_grad=False)
        incidence_A = torch.zeros(n_nodes,n_edges,dtype=torch.float32,device=self.device,requires_grad=False)

        adjacency[connectivity[:,0],connectivity[:,1]] = 1
        adjacency[connectivity[:,1],connectivity[:,0]] = 1
        incidence_A[connectivity[:,0],order] = 1
        incidence_A[connectivity[:,1],order] = 1

        return incidence_A,adjacency

    def global_attention(self,mu,n_nodes):
        if type(n_nodes) is int or n_nodes is None: # normal operation
            mu_global = self.global_attn_NN.Forward(mu,1,None).flatten()
        else: # for mini-batch training
            index_column = torch.zeros(mu.shape[0],dtype=torch.int64,device=self.device)
            for i in range(len(n_nodes)-2):
                index_column[n_nodes[i+1]:] += 1
            index = index_column.tile((self.n_whole_output,1)).T
            mu_global = self.global_attn_NN.Forward(mu,len(n_nodes)-1,index)
        return mu_global

    def mu(self,v,mu,w,mu_global,incidence_A,adjacency,mu_iter,n_nodes):
        '''
        v (array[n_nodes,n_node_input])
        mu(array[n_edges,n_edge_output])
        w (array[n_edges,n_edge_input])
        mu_global(array[n_whole_output] or array[n_graph,n_whole_output])
        '''
        if mu_iter == 0:
            h1 = self.l1_1.forward(v)
            h2 = torch.mm(incidence_A,self.l1_2.forward(w))
            mu = self.ActivationF(h1+h2)
            mu_global = self.global_attention(mu,n_nodes)

        else:
            h3 = self.ActivationF(self.l1_3.forward(mu))
            h4 = self.ActivationF(torch.mm(adjacency,self.l1_4.forward(mu)))
            h5 = self.ActivationF(self.l1_5.forward(mu_global))
            if n_nodes is None: # normal operation
                h5_2 = h5.tile((v.shape[0],1))
            else:
                h5_2 = torch.repeat_interleave(h5,torch.tensor(n_nodes[1:]-n_nodes[:-1],device=self.device),dim=0)
            h345 = torch.concat((h3,h4,h5_2),dim=1)
            mu = self.l1_6.forward(h345)
            mu_global = self.global_attention(mu,n_nodes)

        '''
        NOTE: It might produce better results if regularizing the values by the degree matrix
        '''
           
        return mu, mu_global
        
    def regress(self,mu_global):

        mu_whole = self.ActivationF(self.l2_1(mu_global))
        o = torch.sigmoid(self.l2_2.forward(mu_whole)).flatten()

        return o

    def Forward(self,v,w,connectivity,n_mu_iter=4,nn_batch=None):
       
        '''
        v[n_nodes,n_node_in_features]
        w[n_edges,n_edge_input]
        connectivity[n_edges,2]
        nn_batch[BATCH_SIZE+1] : int
        '''
        IA,D = self.Connectivity(connectivity,v.shape[0])

        if type(v) is np.ndarray: 
            v = torch.tensor(v,dtype=torch.float32,device=self.device,requires_grad=False)
        if type(w) is np.ndarray:
            w = torch.tensor(w,dtype=torch.float32,device=self.device,requires_grad=False)
        mu = torch.zeros((connectivity.shape[0],self.n_node_output),device=self.device)

        if nn_batch is None:
            mu_global = torch.zeros(self.n_whole_output,dtype=torch.float32,device=self.device)
        else:
            mu_global = torch.zeros(len(nn_batch)-1,self.n_whole_output,dtype=torch.float32,device=self.device)

        for i in range(n_mu_iter):
            mu, mu_global = self.mu(v,mu,w,mu_global,IA,D,mu_iter=i,n_nodes=nn_batch)
            # print("iter {0}: {1}".format(i,mu.norm(p=2)))
        
        cl = self.regress(mu_global)

        return cl

    def Save(self,filename,directory=""):
        torch.save(self.to('cpu').state_dict(),os.path.join(directory,filename))
        torch.save(self.global_attn_NN.to('cpu').state_dict(),os.path.join(directory,filename+'_global_attn'))
    
    def Load(self,filename,directory=""):
        self.load_state_dict(torch.load(os.path.join(directory,filename)))
        self.global_attn_NN.load_state_dict(torch.load(os.path.join(directory,filename+'_global_attn')))

class Brain():
    def __init__(self,n_node_inputs,n_edge_inputs,n_node_outputs,whole_graph_feature,use_gpu,validate_ratio=0.1):
        if use_gpu:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.n_node_inputs = n_node_inputs
        self.n_edge_inputs = n_edge_inputs
        self.model = NN(n_node_inputs,n_edge_inputs,n_node_outputs,whole_graph_feature,use_gpu)
        # self.optimizer = torch.optim.SGD(self.model.parameters(),lr=1.0e-1,momentum=0.9)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(),lr=1.0e-3)
        self.memory = deque()
        self.lossfunc = torch.nn.L1Loss()
        self.print_frequency = 10 # print the averaged loss for this number of minibatches
        self.validate_data_ratio = validate_ratio # Use this ratio of the whole training datasets for validation

    def store(self,connectivity,v,w,target):

        v = torch.tensor(v,dtype=torch.float32,device=self.device,requires_grad=False)
        w = torch.tensor(w,dtype=torch.float32,device=self.device,requires_grad=False)
        self.memory.append((connectivity,v,w,target))

    def sample_batch(self,indices):
        batch = [self.memory[index] for index in indices]

        c_batch = np.zeros((0,2),dtype=int)
        v_batch = torch.cat([batch[i][1] for i in range(len(indices))],dim=0)
        w_batch = torch.cat([batch[i][2] for i in range(len(indices))],dim=0)
        target_batch = torch.tensor([batch[i][3] for i in range(len(indices))],dtype=torch.float32,device=self.device,requires_grad=False)
        nn_batch = np.zeros(len(indices)+1,dtype=int)

        nn = 0
        nm = 0
        for i in range(len(indices)):
            c_batch = np.concatenate((c_batch,batch[i][0]+nn),axis=0)
            nn += batch[i][1].shape[0]
            nm += batch[i][2].shape[0]
            nn_batch[i+1] = nn

        return c_batch,v_batch,w_batch,target_batch,nn_batch

    def train(self,n_epoch,shuffle=True):

        n_train_data = int(np.round(len(self.memory)*(1.0-self.validate_data_ratio)))
        
        if random.shuffle:
            all_index = np.random.permutation(len(self.memory))
        else:
            all_index = np.arange(len(self.memory))
        train_data_index = all_index[:n_train_data]
        validation_data_index = all_index[n_train_data:]
            
        least_loss = np.inf
        train_loss_history = np.empty(n_epoch,dtype=float)
        validation_loss_history = np.empty(n_epoch,dtype=float)
        accuracy_history = np.empty(n_epoch+1,dtype=float)

        _, accuracy = self.validate(validation_data_index)
        accuracy_history[0] = accuracy

        for epoch in range(n_epoch):
            print("### Epoch:{0} ###".format(epoch))
            np.random.shuffle(train_data_index)
            train_loss = 0.0
            # running_loss = 0.0
            for i in range(0,n_train_data,BATCH_SIZE):
                c,v,w,target,nm = self.sample_batch(train_data_index[i:i+BATCH_SIZE])
                self.optimizer.zero_grad()
                loss = self.calc_loss(c,v,w,target,nm)
                loss.backward()
                self.optimizer.step()
                # running_loss += loss.item()*len(train_data_index[i:i+BATCH_SIZE])*len(train_data_index[i:i+BATCH_SIZE])
                train_loss += loss.item()*len(train_data_index[i:i+BATCH_SIZE])
                # if i % self.print_frequency == self.print_frequency-1:
                    # print('Minibatch {0}-Loss: {1:.3f}'.format(i+1,running_loss/self.print_frequency))
                    # running_loss = 0.0
            train_loss_history[epoch] = train_loss/n_train_data
            validation_loss, accuracy = self.validate(validation_data_index)
            validation_loss_history[epoch] = validation_loss
            accuracy_history[epoch+1] = accuracy

            if(validation_loss.item() <= least_loss):
                least_loss = validation_loss.item()
                top_scored_epoch = epoch
                top_scored_model = deepcopy(self.model)

        top_scored_model.Save(filename="trained_model")

        with open("result/info.txt", 'w') as f:
            f.write(str.format("top-scored epoch: {0} \n",top_scored_epoch+1))

        with open("result/trainloss.csv", 'w',newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(train_loss_history)
        with open("result/testloss.csv", 'w',newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(validation_loss_history)
        with open("result/accuracy.csv", 'w',newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(accuracy_history)
        return

    def validate(self,validation_data_index,gpu=True):
        loss = 0.0
        tg_all = np.zeros(0)
        diff_all = np.zeros(0)
        for i in range(0,len(validation_data_index),BATCH_SIZE):
            c,v,w,target,nn = self.sample_batch(validation_data_index[i:i+BATCH_SIZE])
            if not gpu:
                v = torch.clone(v.cpu().detach())
                w = torch.clone(w.cpu().detach())
                target = torch.clone(target.cpu().detach())
            loss_temp = self.calc_loss(c,v,w,target,nn)
            loss += loss_temp*len(validation_data_index[i:i+BATCH_SIZE])
            tg_all = np.append(tg_all,target.cpu().detach().numpy())
            diff_all = np.append(diff_all,torch.abs(self.model.Forward(v,w,c,nn_batch=nn)-target).cpu().detach().numpy())
        loss /= len(validation_data_index)
        print('Validation loss: {0:.3f}'.format(loss))
        accuracy = 1-np.var(diff_all)/np.var(tg_all)
        print('Accuracy: {0:.3f}'.format(accuracy))
        return loss, accuracy

    def calc_loss(self,c,v,w,target,nn):
        output = self.model.Forward(v,w,c,nn_batch=nn)
        loss = self.lossfunc(output,target)
        return loss

    # def accuracy(self,c,v,w,target,nn_batch):
    #     output = self.model.Forward(v,w,c,nn_batch=nn_batch)
    #     accuracy = torch.mean(1 - torch.abs(target-output)/target).cpu().detach().numpy()
    #     return accuracy
        

