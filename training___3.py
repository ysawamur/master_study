import numpy as np
import glob
import os
import sys
import time

import pandas as pd

import chainer
from chainer import functions as F
from chainer import links as L
from chainer.datasets import split_dataset_random, get_cross_validation_datasets
from chainer.cuda import to_cpu, to_gpu
from chainer import serializers
import scipy.stats
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


gpu_id = 0
if gpu_id >= 0:
    chainer.cuda.get_device(gpu_id).use()

DATA_SIZE1 = str(input('INPUTDATA_SIZE:'))
DATA_SIZE = str(input('DATA_SIZE:'))
N_DATA = str(input('N_DATA:'))
LOSS = str(input('LOSS_FUNCTION:'))
data = np.load('/work/ysawamura/master_study/simulation/experiment/expmt1/dataset/flux_data_' + DATA_SIZE1 + '.npy')

n_in = 43 # RANGE OF INPUT DATA
n_out = 50 # RANGE OF OUTPUT DATA

n_in_2 = 43-12

min_data = 1e-3
data = data[np.where(np.mean(np.abs(data[:,n_in:n_out]),axis=1)>min_data)]
data = scipy.stats.zscore(data)
frac = 1.0
ndata = data.shape[0]
np.random.seed(seed=30)
inds = np.random.choice(ndata,np.int(ndata*frac),replace=False)
data = data[inds,:]
ndata = data.shape[0]

data = data[:5000000,:]
#test = test[np.where(np.mean(np.abs(test[:,n_in:n_out]),axis=1)>min_data)]
# test = test[np.where(np.mean(np.abs(test[:,[n_in,n_in+1,n_in+2,n_in+3,n_in+5]]),axis=1)>min_data)]

flux =  ['dro','de','dvx','dvy','dvz','dBy','dBz']
flux_1 = ['dro','de','dvx']
flux_2 = ['dvy','dBy']
flux_3 = ['dvz','dBz']


#preprocessing
PREPROCESS = str(input('PREPROCESSING:'))
OPTIMIZER = str(input('OPTIMIZER:'))
day_data = str(input('DAY_DATA:'))






nnode = [130]
nlayer = [4]
nepoch = [2000]
nreport = 100

#Definition Network
class MLP(chainer.Chain):

    def __init__(self, init_node, nodes, initializer = None):
        super(MLP,self).__init__()
        self.init_node = init_node
        self.nlayer = len(nodes)

        with self.init_scope():
            self.add_link('hidlayer1',L.Linear(init_node,nodes[0],initialW=initializer))
            for layer in range(1,self.nlayer-1):
                self.add_link('hidlayer{}'.format(layer+1),L.Linear(None,nodes[layer],initialW=initializer))

        self.add_link('lastlayer',L.Linear(None,nodes[-1],initialW=initializer))

    # Forward operation
    def __call__(self, x):
        u = F.relu(self['hidlayer1'](x))

        for layer in range(1,self.nlayer-1):
            u = F.relu(self['hidlayer{}'.format(layer+1)](u))
        y = self['lastlayer'](u)

        return y


if __name__ == '__main__':
	from pyDOE import *
	nexp = 3
	nfold = 10
	hp = lhs(3,samples=nexp,criterion='cm')
	max_layer = 6
	max_node = n_in*7
	min_layer = 3
	min_node = n_in*1.5
	max_epoch = 10000
	min_epoch = 5000
	#nnode = (min_node+np.array(hp[:,0])*(max_node-min_node)+0.5).astype(np.int32)
	#nlayer = (min_layer+np.array(hp[:,1])*(max_layer-min_layer)+0.5).astype(np.int32)
	#nepoch = (min_epoch+np.array(hp[:,2])*(max_epoch-min_epoch)+0.5).astype(np.int32)
	nnode0 = [172]
	nnode1 = [100]
	nnode2 = [100]
	nlayer = [4]
	nepoch = [2000]
	nreport = 100

    #divide data about train and validation
	data_list = get_cross_validation_datasets(data,nfold)

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
	    nodes2 = []
	    nodes3 = []
	    for l in range(nlayer[it]):
	#        nodes.append(nnode[it])
		      #nodes1.append(np.max([np.int(nnode0[it]/2**l+0.5),7]))
		      #nodes2.append(np.max([np.int(nnode1[it]/2**l+0.5),7]))
		      #nodes3.append(np.max([np.int(nnode2[it]/2**l+0.5),7]))
		      nodes1.append(np.max([np.int(nnode0[it]),3]))
		      nodes2.append(np.max([np.int(nnode1[it]),2]))
		      nodes3.append(np.max([np.int(nnode2[it]),2]))

	    nodes1.append(3)
	    nodes2.append(2)
	    nodes3.append(2)
	    print(it,nodes1,nodes2,nodes3, nlayer[it],nepoch[it])
	    NODES = str(nodes1[1])
        #training roop
	    for itry in range(1):
    		#train, valid = data_list[itry]
    		train, valid = data[:len(data)-100000,:],data[len(data)-100000:,:]
    		train = np.array(train)
    		valid = np.array(valid)

    		ntrain = train.shape[0]
    		nvalid = valid.shape[0]

    		train1 = np.delete(train, [46,47,48,49], 1)
    		valid1 = np.delete(valid, [46,47,48,49], 1)
    		train2 = np.delete(train, [6,7,8,9,10,11,24,25,26,27,28,29,37,38,39,40,41,42,43,44,45,47,49], 1)
    		valid2 = np.delete(valid, [6,7,8,9,10,11,24,25,26,27,28,29,37,38,39,40,41,42,43,44,45,47,49], 1)
    		train3 = np.delete(train, [6,7,8,9,10,11,18,19,20,21,22,23,31,32,33,34,35,36,43,44,45,46,48], 1)
    		valid3 = np.delete(valid, [6,7,8,9,10,11,18,19,20,21,22,23,31,32,33,34,35,36,43,44,45,46,48], 1)

    		if PREPROCESS == 'Stdztn':
    		    train_1 = train1[:,:43]
    		    train_1_label = train1[:,43:46]
    		    train_2 = train2[:,:25]
    		    train_2_label = train2[:,25:27]
    		    train_3 = train3[:,:25]
    		    train_3_label = train3[:,25:27]
    		    valid_1 = valid1[:,:43]
    		    valid_1_label = valid1[:,43:46]
    		    valid_2 = valid2[:,:25]
    		    valid_2_label = valid2[:,25:27]
    		    valid_3 = valid3[:,:25]
    		    valid_3_label = valid3[:,25:27]
    		elif PREPROCESS == 'Nmlztn':
    		    mn = preprocessing.MinMaxScaler()
    		    data = mn.fit_transform(data)


    		nbatch = np.int(ntrain/20)

    		model_1 = MLP(43,nodes1,initializer=chainer.initializers.HeNormal())
    		model_2 = MLP(25,nodes2,initializer=chainer.initializers.HeNormal())
    		model_3 = MLP(25,nodes3,initializer=chainer.initializers.HeNormal())


            #send to gpu
    		if gpu_id >= 0:
    		    model_1.to_gpu(gpu_id)
    		    model_2.to_gpu(gpu_id)
    		    model_3.to_gpu(gpu_id)
    		    train_1 = to_gpu(train_1)
    		    valid_1  = to_gpu(valid_1)
    		    train_2 = to_gpu(train_2)
    		    valid_2  = to_gpu(valid_2)
    		    train_3 = to_gpu(train_3)
    		    valid_3  = to_gpu(valid_3)
    		    train_1_label = to_gpu(train_1_label)
    		    valid_1_label  = to_gpu(valid_1_label)
    		    train_2_label = to_gpu(train_2_label)
    		    valid_2_label  = to_gpu(valid_2_label)
    		    train_3_label = to_gpu(train_3_label)
    		    valid_3_label  = to_gpu(valid_3_label)



            #definition optimizer
    		if OPTIMIZER == 'Adam':
	    		optimizer_1 = chainer.optimizers.Adam()
	    		optimizer_2 = chainer.optimizers.Adam()
	    		optimizer_3 = chainer.optimizers.Adam()
    		elif OPTIMIZER == 'AMSBound':
	    		optimizer_1 = chainer.optimizers.AMSBound()
	    		optimizer_2 = chainer.optimizers.AMSBound()
	    		optimizer_3 = chainer.optimizers.AMSBound()
    		optimizer_1.use_cleargrads()
    		optimizer_2.use_cleargrads()
    		optimizer_3.use_cleargrads()
    		optimizer_1.setup(model_1)
    		optimizer_2.setup(model_2)
    		optimizer_3.setup(model_3)




    		for i in range(nepoch[it]):

    		    etime0 = time.time()
    		    icount = 0
    		    inds = np.random.choice(ntrain,ntrain,replace=False)
    		    while icount*nbatch <= ntrain-nbatch:

    		        # SET MINI-BATCH
    		        x_batch_1 = train_1[inds[icount*nbatch:(icount+1)*nbatch]]
    		        t_batch_1 = train_1_label[inds[icount*nbatch:(icount+1)*nbatch]]
    		        x_batch_2 = train_2[inds[icount*nbatch:(icount+1)*nbatch]]
    		        t_batch_2 = train_2_label[inds[icount*nbatch:(icount+1)*nbatch]]
    		        x_batch_3 = train_3[inds[icount*nbatch:(icount+1)*nbatch]]
    		        t_batch_3 = train_3_label[inds[icount*nbatch:(icount+1)*nbatch]]

    		        # FORWARD PROP.
    		        y_1 = model_1(x_batch_1)
    		        y_2 = model_2(x_batch_2)
    		        y_3 = model_3(x_batch_3)
    		        if LOSS == 'RMSE':
    		            loss_1 = F.mean_squared_error(y_1,t_batch_1)
    		            loss_2 = F.mean_squared_error(y_2,t_batch_2)
    		            loss_3 = F.mean_squared_error(y_3,t_batch_3)
    		        elif LOSS == 'MAE':
    		            loss_1 = F.mean_absolute_error(y_1,t_batch_1)
    		            loss_2 = F.mean_absolute_error(y_2,t_batch_2)
    		            loss_3 = F.mean_absolute_error(y_3,t_batch_3)

    		        # BACK PROP.
    		        model_1.cleargrads()
    		        loss_1.backward()

    		        model_2.cleargrads()
    		        loss_2.backward()

    		        model_3.cleargrads()
    		        loss_3.backward()

    		        # UPDATE
    		        optimizer_1.update()
    		        optimizer_2.update()
    		        optimizer_3.update()

    		        icount += 1

    		    etime1 = time.time()


    		    if i%nreport == 0:
    		        with chainer.using_config('enable_backprop', False):
    		            print('Exp:',it,'  Itry:',itry,'  Epoch:',i)
    		            for j in range(3):

                            #predict model1(ro,e,vx,By,Bz)
    		                predict_train_1 = model_1(train_1[:,0:n_in])
    		                predict_valid_1 = model_1(valid_1[:,0:n_in])
    		                if LOSS == 'RMSE':
    		                    loss_train_tmp = to_cpu(F.mean_squared_error(predict_train_1[:,j],train_1_label[:,j]).data)
    		                    loss_valid_tmp = to_cpu(F.mean_squared_error(predict_valid_1[:,j],valid_1_label[:,j]).data)
    		                elif LOSS == 'MAE':
    		                    loss_train_tmp = to_cpu(F.mean_absolute_error(predict_train_1[:,j],train_1_label[:,j]).data)
    		                    loss_valid_tmp = to_cpu(F.mean_absolute_error(predict_valid_1[:,j],valid_1_label[:,j]).data)
    		                it_train_loss.append(loss_train_tmp)
    		                it_valid_loss.append(loss_valid_tmp)
    		                r2_train_tmp = to_cpu(F.r2_score(predict_train_1[:,j],train_1_label[:,j]).data)
    		                it_train_r2.append(r2_train_tmp)
    		                r2_valid_tmp = to_cpu(F.r2_score(predict_valid_1[:,j],valid_1_label[:,j]).data)
    		                it_valid_r2.append(r2_valid_tmp)
    		                print(flux_1[j],' R2 score (train):',r2_train_tmp,'  R2 score (valid):',r2_valid_tmp)
    		                print('loss(train):',loss_train_tmp,'loss(valid):',loss_valid_tmp)
    		            for j in range(2):

    		                predict_train_2 = model_2(train_2[:,0:25])
    		                predict_valid_2 = model_2(valid_2[:,0:25])
    		                if LOSS == 'RMSE':
    		                    loss_train_tmp = to_cpu(F.mean_squared_error(predict_train_2[:,j],train_2_label[:,j]).data)
    		                    loss_valid_tmp = to_cpu(F.mean_squared_error(predict_valid_2[:,j],valid_2_label[:,j]).data)
    		                elif LOSS == 'MAE':
    		                    loss_train_tmp = to_cpu(F.mean_absolute_error(predict_train_2[:,j],train_2_label[:,j]).data)
    		                    loss_valid_tmp = to_cpu(F.mean_absolute_error(predict_valid_2[:,j],valid_2_label[:,j]).data)
    		                it_train_loss.append(loss_train_tmp)
    		                it_valid_loss.append(loss_valid_tmp)
    		                r2_train_tmp = to_cpu(F.r2_score(predict_train_2[:,j],train_2_label[:,j]).data)
    		                it_train_r2.append(r2_train_tmp)
    		                r2_valid_tmp = to_cpu(F.r2_score(predict_valid_2[:,j],valid_2_label[:,j]).data)
    		                it_valid_r2.append(r2_valid_tmp)
    		                print(flux_2[j],' R2 score (train):',r2_train_tmp,'  R2 score (valid):',r2_valid_tmp)
    		                print('loss(train):',loss_train_tmp,'loss(valid):',loss_valid_tmp)

    		            for j in range(2):

    		                predict_train_3 = model_3(train_3[:,0:25])
    		                predict_valid_3 = model_3(valid_3[:,0:25])
    		                if LOSS == 'RMSE':
    		                    loss_train_tmp = to_cpu(F.mean_squared_error(predict_train_3[:,j],train_3_label[:,j]).data)
    		                    loss_valid_tmp = to_cpu(F.mean_squared_error(predict_valid_3[:,j],valid_3_label[:,j]).data)
    		                elif LOSS == 'MAE':
    		                    loss_train_tmp = to_cpu(F.mean_absolute_error(predict_train_3[:,j],train_3_label[:,j]).data)
    		                    loss_valid_tmp = to_cpu(F.mean_absolute_error(predict_valid_3[:,j],valid_3_label[:,j]).data)
    		                it_train_loss.append(loss_train_tmp)
    		                it_valid_loss.append(loss_valid_tmp)
    		                r2_train_tmp = to_cpu(F.r2_score(predict_train_3[:,j],train_3_label[:,j]).data)
    		                it_train_r2.append(r2_train_tmp)
    		                r2_valid_tmp = to_cpu(F.r2_score(predict_valid_3[:,j],valid_3_label[:,j]).data)
    		                it_valid_r2.append(r2_valid_tmp)
    		                print(flux_3[j],' R2 score (train):',r2_train_tmp,'  R2 score (valid):',r2_valid_tmp)
    		                print('loss(train):',loss_train_tmp,'loss(valid):',loss_valid_tmp)



    		with chainer.using_config('enable_backprop', False):

    		    loss_train.append(it_train_loss)
    		    loss_valid.append(it_valid_loss)
    		    R2_train.append(it_train_r2)
    		    R2_valid.append(it_valid_r2)
    		    model_1.to_cpu()
    		    model_2.to_cpu()
    		    model_3.to_cpu()
    		    data_1 = np.delete(data, [46,47,48,49], 1)
    		    data_2 = np.delete(data, [6,7,8,9,10,11,24,25,26,27,28,29,37,38,39,40,41,42,43,44,45,47,49], 1)
    		    data_3 = np.delete(data, [6,7,8,9,10,11,18,19,20,21,22,23,31,32,33,34,35,36,43,44,45,46,48], 1)
    		    y_1_p = model_1(to_cpu(valid1[:,0:n_in]))
    		    y_2_p = model_2(to_cpu(valid2[:,0:25]))
    		    y_3_p = model_3(to_cpu(valid3[:,0:25]))
    		    os.system('mkdir ./vary_input2/'+DATA_SIZE+'/'+LOSS+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+'n_data'+N_DATA)
    		    np.save('./vary_input2/'+DATA_SIZE+'/'+LOSS+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+'n_data'+N_DATA+'/valid1.npy', valid1)
    		    np.save('./vary_input2/'+DATA_SIZE+'/'+LOSS+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+'n_data'+N_DATA+'/valid2.npy', valid2)
    		    np.save('./vary_input2/'+DATA_SIZE+'/'+LOSS+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+'n_data'+N_DATA+'/valid3.npy', valid3)
    		    np.save('./vary_input2/'+DATA_SIZE+'/'+LOSS+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+'n_data'+N_DATA+'/data_predict1_fix' + '{:02d}'.format(it) + '.npy', y_1_p.data)
    		    np.save('./vary_input2/'+DATA_SIZE+'/'+LOSS+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+'n_data'+N_DATA+'/data_predict2_fix' + '{:02d}'.format(it) + '.npy', y_2_p.data)
    		    np.save('./vary_input2/'+DATA_SIZE+'/'+LOSS+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+'n_data'+N_DATA+'/data_predict3_fix' + '{:02d}'.format(it) + '.npy', y_3_p.data)
    		    serializers.save_npz('./vary_input2/'+DATA_SIZE+'/'+LOSS+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+'n_data'+N_DATA+'/model_1' + '{:03d}'.format(it) + '.npz', model_1)
    		    serializers.save_npz('./vary_input2/'+DATA_SIZE+'/'+LOSS+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+'n_data'+N_DATA+'/model_2' + '{:03d}'.format(it) + '.npz', model_2)
    		    serializers.save_npz('./vary_input2/'+DATA_SIZE+'/'+LOSS+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+'n_data'+N_DATA+'/model_3' + '{:03d}'.format(it) + '.npz', model_3)
    		    it_train_loss =[]
    		    it_valid_loss = []
    		    it_train_r2 = []
    		    it_valid_r2 = []

	length = len(R2_train)
	x_int = int(len(R2_train[0])/7)
	x = np.arange(0,x_int)
	R2_train = np.array(R2_train[0]).reshape(x_int, 7)
	R2_valid = np.array(R2_valid[0]).reshape(x_int, 7)
	np.save('./vary_input2/'+DATA_SIZE+'/'+LOSS+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+'n_data'+N_DATA+'/nlayer_net2',nlayer)
	np.save('./vary_input2/'+DATA_SIZE+'/'+LOSS+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+'n_data'+N_DATA+'/nnode_net2',nnode)
	np.save('./vary_input2/'+DATA_SIZE+'/'+LOSS+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+'n_data'+N_DATA+'/nepoch_net2',nepoch)
	np.save('./vary_input2/'+DATA_SIZE+'/'+LOSS+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+'n_data'+N_DATA+'/R2_train',R2_train)
	np.save('./vary_input2/'+DATA_SIZE+'/'+LOSS+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+'n_data'+N_DATA+'/R2_valid',R2_valid)
	np.save('./vary_input2/'+DATA_SIZE+'/'+LOSS+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+'n_data'+N_DATA+'/loss_train',loss_train)
	np.save('./vary_input2/'+DATA_SIZE+'/'+LOSS+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+'n_data'+N_DATA+'/loss_valid',loss_valid)
	simulation_data = pd.DataFrame({"day": day_data,"net_work":"3_network", "input":"same", "Network1":nodes1[0], "Network2":nodes2[0], "Network3":nodes3[0], "R2_score_dro":R2_valid[x_int-1,0], "R2_score_de":R2_valid[x_int-1,1], "R2_score_dvx":R2_valid[x_int-1,2], "R2_score_dvy":R2_valid[x_int-1,3], "R2_score_dvz":R2_valid[x_int-1,4], "R2_score_dBy":R2_valid[x_int-1,5], "R2_score_dBz":R2_valid[x_int-1,6],"DATA_SIZE":N_DATA}, index=['i',])
	simulation_data.to_csv('simulation_data.csv', mode='a', header=False)
	sys.exit()
