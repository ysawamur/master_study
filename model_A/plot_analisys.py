import numpy as np
import matplotlib.pyplot as plt


DATA_SIZE = str(input('DATA_SIZE:'))
LOSS_FUNCTION = str(input('LOSS_FUNCTION:'))
PREPROCESS = str(input('PREPROCESSING:'))
OPTIMIZER = str(input('OPTIMIZER:'))
N_DATA = str(input('N_DATA:'))

#INPUT_DATA = str(input('INPUT_DATA:'))

R2_train = np.load('./'+ DATA_SIZE+'/'+LOSS_FUNCTION+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+N_DATA+'/R2_train.npy', allow_pickle=True)
R2_valid = np.load('./'+ DATA_SIZE + '/'+LOSS_FUNCTION+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+N_DATA+'/R2_valid.npy', allow_pickle=True)
node = np.load('./'+DATA_SIZE +'/'+LOSS_FUNCTION+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+N_DATA+'/nnode_net2.npy')


'''
for l in range(len(R2_train_ld)):
	length = len(R2_train_ld)
	x_int = int(len(R2_train_ld[l])/7)
	x = np.arange(0,x_int)
	R2_train = np.array(R2_train_ld[l]).reshape(x_int, 7)
	R2_valid = np.array(R2_valid_ld[l]).reshape(x_int, 7)

	fig = plt.figure(figsize=(15.0, 10.0))
	plt.subplot(1,2,1)
	plt.ylim([0.75,1])
	plt.plot(x, R2_train[:,0], label='dro')
	plt.plot(x, R2_train[:,1], label='de')
	plt.plot(x, R2_train[:,2], label='dvx')
	plt.plot(x, R2_train[:,3], label='dvy')
	plt.plot(x, R2_train[:,4], label='dvz')
	plt.plot(x, R2_train[:,5], label='dby')
	plt.plot(x, R2_train[:,6], label='dbz')
	plt.title('R2_score_train')
	plt.xlabel('epoch')
	plt.ylabel('R2_score')
	plt.legend(loc='lower right')

	plt.subplot(1,2,2)
	plt.ylim([0.75,1])
	plt.plot(x, R2_valid[:,0], label='dro')
	plt.plot(x, R2_valid[:,1], label='de')
	plt.plot(x, R2_valid[:,2], label='dvx')
	plt.plot(x, R2_valid[:,3], label='dvy')
	plt.plot(x, R2_valid[:,4], label='dvz')
	plt.plot(x, R2_valid[:,5], label='dby')
	plt.plot(x, R2_valid[:,6], label='dbz')
	plt.title('R2_score_valid')
	plt.xlabel('epoch')
	plt.ylabel('R2_score')
	plt.legend(loc='lower right')
'''
length = len(R2_train)
x_int = int(len(R2_train))
x = np.arange(0,x_int)

fig, ax = plt.subplots(2, 4, figsize=(15,6))
ax[0][0].plot(x,R2_train[:,0])
ax[0][0].plot(x,R2_valid[:,0])
ax[0][1].plot(x,R2_train[:,1])
ax[0][1].plot(x,R2_valid[:,1])
ax[0][2].plot(x,R2_train[:,2])
ax[0][2].plot(x,R2_valid[:,2])
ax[0][3].plot(x,R2_train[:,3])
ax[0][3].plot(x,R2_valid[:,3])
ax[1][0].plot(x,R2_train[:,4])
ax[1][0].plot(x,R2_valid[:,4])
ax[1][1].plot(x,R2_train[:,5])
ax[1][1].plot(x,R2_valid[:,5])
ax[1][2].plot(x,R2_train[:,6])
ax[1][2].plot(x,R2_valid[:,6])

ax[0][0].set_xlabel("epoch")
ax[0][0].set_ylabel("R2_score")
ax[0][1].set_xlabel("epoch")
ax[0][1].set_ylabel("R2_score")
ax[0][2].set_xlabel("epoch")
ax[0][2].set_ylabel("R2_score")
ax[0][3].set_xlabel("epoch")
ax[0][3].set_ylabel("Predit")
ax[1][0].set_xlabel("epoch")
ax[1][0].set_ylabel("R2_score")
ax[1][1].set_xlabel("epoch")
ax[1][1].set_ylabel("R2_score")
ax[1][2].set_xlabel("epoch")
ax[1][2].set_ylabel("R2_score")

ax[0][0].set_ylim([0.9, 1.0])
ax[0][1].set_ylim([0.9, 1.0])
ax[0][2].set_ylim([0.9, 1.0])
ax[0][3].set_ylim([0.9, 1.0])
ax[1][0].set_ylim([0.9, 1.0])
ax[1][1].set_ylim([0.9, 1.0])
ax[1][2].set_ylim([0.9, 1.0])

ax[0][0].set_title("ro")
ax[0][1].set_title("e")
ax[0][2].set_title("vx")
ax[0][3].set_title("vy")
ax[1][0].set_title("vz")
ax[1][1].set_title("By")
ax[1][2].set_title("Bz")

ax[0][0].legend(['train','valid'], loc='lower right')
ax[0][1].legend(['train','valid'], loc='lower right')
ax[0][2].legend(['train','valid'], loc='lower right')
ax[0][3].legend(['train','valid'], loc='lower right')
ax[1][0].legend(['train','valid'], loc='lower right')
ax[1][1].legend(['train','valid'], loc='lower right')
ax[1][2].legend(['train','valid'], loc='lower right')

ax[1][3].axis("off")

plt.tight_layout()
plt.savefig('./'+ DATA_SIZE + '/'+LOSS_FUNCTION+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+N_DATA+'/R2_score.png')
plt.close()

	
