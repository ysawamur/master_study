import numpy as np
import glob
import os
import sys
import time

import chainer
from chainer import functions as F
from chainer import links as L
from chainer.datasets import split_dataset_random, get_cross_validation_datasets
from chainer.cuda import to_cpu, to_gpu
from chainer import serializers

import pandas as pd
import scipy.stats
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

#setting GPU id
gpu_id = 0
if gpu_id >= 0:
    chainer.cuda.get_device(gpu_id).use()

#loading training data
TRANING_DATASIZE = str(input('TRANING_DATASIZE:'))
data = np.load('/work/ysawamura/master_study/simulation/experiment/expmt1/dataset/flux_data_' + TRANING_DATASIZE + '.npy')

#select save directly
SAVE_DIR = str(input('SAVE_DIR:'))
#select loss function
LOSS = str(input('LOSS_FUNCTION:'))
#select preprocessing function
PREPROCESS = str(input('PREPROCESSING:'))
#select optimizer
OPTIMIZER = str(input('OPTIMIZER:'))
#select node size
N_DATA = str(input('N_DATA:'))
#day information
day_data = str(input('DAY_DATA:'))

#range input data
n_in = 43
#range output data
n_out = 50

#data preprocessing
min_data = 1e-3
data = data[np.where(np.mean(np.abs(data[:,n_in:n_out]),axis=1)>min_data)]
print(data.shape)
x_mean = np.mean(data[:,:43], axis=0)
x_std = np.std(data[:,:43], axis=0)
t_mean = np.mean(data[:,43:50], axis=0)
t_std = np.std(data[:,43:50], axis=0)
data = scipy.stats.zscore(data)

#mixing training data
frac = 1.0
ndata = data.shape[0]
np.random.seed(seed=30)
inds = np.random.choice(ndata,np.int(ndata*frac),replace=False)
data = data[inds,:]
ndata = data.shape[0]

#range of using data
data = data[:6000000,:]
print(data.shape)








#nural network
class MLP(chainer.Chain):

    def __init__(self, nodes, initializer = None):
        super(MLP,self).__init__()

        self.nlayer = len(nodes)

        with self.init_scope():
            self.add_link('hidlayer1',L.Linear(43,nodes[0],initialW=initializer))
            for layer in range(1,self.nlayer-1):
                self.add_link('hidlayer{}'.format(layer+1),L.Linear(None,nodes[layer],initialW=initializer))

        self.add_link('lastlayer',L.Linear(None,nodes[-1],initialW=initializer))

    #forward operation
    def __call__(self, x):
        u = F.relu(self['hidlayer1'](x))

        for layer in range(1,self.nlayer-1):
            u = F.relu(self['hidlayer{}'.format(layer+1)](u))
        y = self['lastlayer'](u)

        return y
#output data tag
flux =  ['dro','de','dvx','dvy','dvz','dBy','dBz']
flux_1 = ['dro','de','dvx','dBy','dBz']
flux_2 = ['dvy']
flux_3 = ['dvz']

#training session
if __name__ == '__main__':
	from pyDOE import *
	nexp = 5
	nfold = 10
	hp = lhs(3,samples=nexp,criterion='cm')
	max_layer = 5
	max_node = n_in*7
	min_layer = 2
	min_node = n_in/2
	max_epoch = 20000
	min_epoch = 1000
	#nnode = (min_node+np.array(hp[:,0])*(max_node-min_node)+0.5).astype(np.int32)
	#nlayer = (min_layer+np.array(hp[:,1])*(max_layer-min_layer)+0.5).astype(np.int32)
	#nepoch = (min_epoch+np.array(hp[:,2])*(max_epoch-min_epoch)+0.5).astype(np.int32)
	nnode = [172]
	nlayer = [4]
	nepoch = [2000]
	nreport = 100

	#data_list = get_cross_validation_datasets(data,nfold)

	r2_train_1 = np.zeros(nexp)
	r2_valid_1 = np.zeros(nexp)
	loss_train =[]
	loss_valid = []
	R2_train = []
	R2_valid = []
	it_train_loss =[]
	it_valid_loss = []
	it_train_r2 = []
	it_valid_r2 = []

	r2_train_2 = np.zeros(nexp)
	r2_valid_2 = np.zeros(nexp)


	for it in range(1):

	    nodes1 = []
	    nodes2 = 0
	    nodes3 = 0

	    for l in range(nlayer[it]):
	        nodes1.append(nnode[it])
	        #nodes1.append(np.max([np.int(nnode[it]/2**l+0.5),7]))

	    nodes1.append(7)
	    print(it,nodes1,nlayer[it],nepoch[it])
        #training roop
	    for itry in range(1):
    		train, valid = data[:len(data)-100000,:],data[len(data)-100000:,:]
    		train_1 = np.array(train)
    		valid_1 = np.array(valid)

    		ntrain = train_1.shape[0]
    		print("ntrain:",ntrain, "valid:" ,valid_1.shape)
    		nvalid = valid_1.shape[0]

    		nbatch = np.int(ntrain/20)
    		print(nbatch)

    		model_1 = MLP(nodes1,initializer=chainer.initializers.HeNormal())

            #send to gpu
    		if gpu_id >= 0:
    		    model_1.to_gpu(gpu_id)
    		    train_1 = to_gpu(train_1)
    		    valid_1  = to_gpu(valid_1)

            #definition optimizer
    		if OPTIMIZER == 'Adam':
	    		optimizer_1 = chainer.optimizers.Adam()
    		elif OPTIMIZER == 'AMSBound':
	    		optimizer_1 = chainer.optimizers.AMSBound()
    		optimizer_1.use_cleargrads()
    		optimizer_1.setup(model_1)




    		for i in range(nepoch[it]):

    		    etime0 = time.time()
    		    icount = 0
    		    inds = np.random.choice(ntrain,ntrain,replace=False)
    		    while icount*nbatch <= ntrain-nbatch:

    		        # set minibatch
    		        x_batch_1 = train_1[inds[icount*nbatch:(icount+1)*nbatch],0:n_in]
    		        t_batch_1 = train_1[inds[icount*nbatch:(icount+1)*nbatch],n_in:n_out]


    		        #forward prop
    		        y_1 = model_1(x_batch_1)
    		        if LOSS == 'RMSE':
    		            loss_1 = F.mean_squared_error(y_1,t_batch_1)
    		        elif LOSS == 'MAE':
    		            loss_1 = F.mean_absolute_error(y_1,t_batch_1)

    		        #back prop
    		        model_1.cleargrads()
    		        loss_1.backward()


    		        #update
    		        optimizer_1.update()
    		        icount += 1

    		    etime1 = time.time()


    		    if i%nreport == 0:
    		        with chainer.using_config('enable_backprop', False):
    		            print('Exp:',it,'  Itry:',itry,'  Epoch:',i)
    		            for j in range(7):

                            #predict model1(ro,e,vx,By,Bz)
    		                predict_train_1 = model_1(train_1[:,0:n_in])
    		                predict_valid_1 = model_1(valid_1[:,0:n_in])
                            #define loss function
    		                if LOSS == 'RMSE':
    		                    loss_train_tmp = to_cpu(F.mean_squared_error(predict_train_1[:,j],train_1[:,n_in + j]).data)
    		                    loss_valid_tmp = to_cpu(F.mean_squared_error(predict_valid_1[:,j],valid_1[:,n_in + j]).data)
    		                elif LOSS == 'MAE':
    		                    loss_train_tmp = to_cpu(F.mean_absolute_error(predict_train_1[:,j],train_1[:,n_in + j]).data)
    		                    loss_valid_tmp = to_cpu(F.mean_absolute_error(predict_valid_1[:,j],valid_1[:,n_in + j]).data)
    		                it_train_loss.append(loss_train_tmp)
    		                it_valid_loss.append(loss_valid_tmp)
    		                r2_train_tmp = to_cpu(F.r2_score(predict_train_1[:,j],train_1[:,n_in + j]).data)
    		                it_train_r2.append(r2_train_tmp)
    		                r2_valid_tmp = to_cpu(F.r2_score(predict_valid_1[:,j],valid_1[:,n_in + j]).data)
    		                it_valid_r2.append(r2_valid_tmp)
    		                print(flux[j],' R2 score (train):',r2_train_tmp,'  R2 score (valid):',r2_valid_tmp)
    		                print('loss(train):',loss_train_tmp,'loss(valid):',loss_valid_tmp)


    		with chainer.using_config('enable_backprop', False):
    		    	loss_train.append(it_train_loss)
    		    	loss_valid.append(it_valid_loss)
    		    	R2_train.append(it_train_r2)
    		    	R2_valid.append(it_valid_r2)
    		    	model_1.to_cpu()
    		    	y_1_p = model_1(to_cpu(valid_1[:,0:n_in]))
    		    	print(y_1_p.shape)
    		    	os.system('mkdir ./'+SAVE_DIR+'/'+LOSS+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+'n_data'+N_DATA)
    		    	np.save('./'+SAVE_DIR+'/'+LOSS+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+'n_data'+N_DATA+'/data_predict1_fix' + '{:02d}'.format(it) + '.npy', y_1_p.data)
    		    	np.save('./'+SAVE_DIR+'/'+LOSS+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+'n_data'+N_DATA+'/valid.npy', valid_1)
    		    	serializers.save_npz('./'+SAVE_DIR+'/'+LOSS+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+'n_data'+N_DATA+'/model_1' + '{:03d}'.format(it) + '.npz', model_1)
    		    	it_train_loss =[]
    		    	it_valid_loss = []
    		    	it_train_r2 = []
    		    	it_valid_r2 = []

	#save training data
	length = len(R2_train)
	x_int = int(len(R2_train[0])/7)
	x = np.arange(0,x_int)
	R2_train = np.array(R2_train[0]).reshape(x_int, 7)
	R2_valid = np.array(R2_valid[0]).reshape(x_int, 7)
	np.save('./'+SAVE_DIR+'/'+LOSS+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+'n_data'+N_DATA+'/nlayer_net2',nlayer)
	np.save('./'+SAVE_DIR+'/'+LOSS+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+'n_data'+N_DATA+'/nnode_net2',nnode)
	np.save('./'+SAVE_DIR+'/'+LOSS+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+'n_data'+N_DATA+'/nepoch_net2',nepoch)
	np.save('./'+SAVE_DIR+'/'+LOSS+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+'n_data'+N_DATA+'/R2_train',R2_train)
	np.save('./'+SAVE_DIR+'/'+LOSS+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+'n_data'+N_DATA+'/R2_valid',R2_valid)
	np.save('./'+SAVE_DIR+'/'+LOSS+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+'n_data'+N_DATA+'/loss_train',loss_train)
	np.save('./'+SAVE_DIR+'/'+LOSS+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+'n_data'+N_DATA+'/loss_valid',loss_valid)
	simulation_data = pd.DataFrame({"day": day_data,"net_work":"1_network", "input":"same", "Network1":nodes1[0], \n
                                "Network2":nodes2, "Network3":nodes3,"R2_score_dro":R2_valid[x_int-1,0], "R2_score_de":R2_valid[x_int-1,1], \n
                                "R2_score_dvx":R2_valid[x_int-1,2], "R2_score_dvy":R2_valid[x_int-1,3], "R2_score_dvz":R2_valid[x_int-1,4], \n
                                "R2_score_dBy":R2_valid[x_int-1,5],"R2_score_dBz":R2_valid[x_int-1,6],"DATASIZE":N_DATA}, index=['i',])
	simulation_data.to_csv('../simulation_data.csv' ,mode='a')

    #save network data
	from scipy.io import FortranFile
	f = FortranFile('./'+SAVE_DIR+'/'+LOSS+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+'n_data'+N_DATA+'/nn.dat', 'w')
	f.write_record(nlayer[0])
	f.write_record(nnode[0])
	f.write_record(x_mean)
	f.write_record(x_std)
	f.write_record(t_mean)
	f.write_record(t_std)
	for pp in model_1.params():
	    f.write_record(to_cpu(pp.data))
	    print(pp.data)
	f.close()
	sys.exit()
