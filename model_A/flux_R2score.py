import numpy as np
import matplotlib.pyplot as plt

import chainer
from chainer import functions as F

import scipy.stats
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

DATA_SIZE1 = str(input('INPUTDATA_SIZE:'))
DATA_SIZE = str(input('DATA_SIZE:'))
LOSS_FUNCTION = str(input('LOSS_FUNCTION:'))
EXPMT = str(input('EXPMT:'))
NET_WORK = str(input('NET_WORK:'))
PREPROCESS = str(input('PREPROCESSING:'))
N_DATA = str(input('N_DATA:'))
OPTIMIZER = 'Adam'
data = np.load('/work/ysawamura/master_study/simulation/experiment/expmt1/dataset/flux_data_' + DATA_SIZE1 + '.npy')

n_in = 43 # RANGE OF INPUT DATA
n_out = 50 # RANGE OF OUTPUT DATA

n_in_2 = 43-12

min_data = 1e-3
data = data[np.where(np.mean(np.abs(data[:,n_in:n_out]),axis=1)>min_data)]
STD = np.std(data,axis=0)
AVE = np.mean(data, axis=0)
valid = np.load('./'+ DATA_SIZE + '/'+LOSS_FUNCTION+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+N_DATA+'/valid.npy')

predict1 = np.load('./'+ DATA_SIZE + '/'+LOSS_FUNCTION+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+N_DATA+'/data_predict1_fix00.npy')

valid = valid*STD + AVE
predict_1 = predict1*STD[43:50] + AVE[43:50]


flux =  ['ro','e','vx','vy','vz','By','Bz']
flux_1 = ['dro','de','dvx','vdy','dvz','dBy','dBz']
flux_2 = ['dvy']
flux_3 = ['dvz']

data_f = valid[:, 43:50]
f0 = valid[:,50:]
T_flux = data_f + f0
P_flux = predict_1 + f0
x = np.arange(len(T_flux))
ro = T_flux[:,0]
e = T_flux[:,1]
vx = T_flux[:,2]
vy = T_flux[:,3]
vz = T_flux[:,4]
By = T_flux[:,5]
Bz = T_flux[:,6]

predict_ro = P_flux[:,0]
predict_e = P_flux[:,1]
predict_vx = P_flux[:,2]
predict_vy = P_flux[:,3]
predict_vz = P_flux[:,4]
predict_By = P_flux[:,5]
predict_Bz = P_flux[:,6]

'''
predict_ro = predict_1[:,0] + f0[:,0]
predict_e = predict_1[:,1] + f0[:,1]

predict_vx = predict_1[:,2] + f0[:,2]

predict_vy = predict_1[:,3] + f0[:,3]

predict_vz = predict_1[:,4] + f0[:,4]

predict_By = predict_1[:,5] + f0[:,5]

predict_Bz = predict_1[:,6] + f0[:,6]
'''

R2 = [F.r2_score(predict_ro,ro), F.r2_score(predict_e,e), F.r2_score(predict_vx,vx), F.r2_score(predict_vy,vy),F.r2_score(predict_vz,vz),F.r2_score(predict_By,By),F.r2_score(predict_Bz,Bz)]

R2_df = [F.r2_score(predict_1[:,0],data_f[:,0]), F.r2_score(predict_1[:,1],data_f[:,1]), F.r2_score(predict_1[:,2],data_f[:,2]),\
		F.r2_score(predict_1[:,3],data_f[:,3]),F.r2_score(predict_1[:,4],data_f[:,4]),F.r2_score(predict_1[:,5],data_f[:,5]),F.r2_score(predict_1[:,6],data_f[:,6])]
for i in range(len(R2)):
    print(flux[i],' R2 score :',R2[i])
    print(flux_1[i],' R2 score :',R2_df[i])
    print(T_flux[i])
    print(P_flux[i])

kwargs = dict(c='deeppink',s=4, alpha=0.5,zorder=1, label='Predict')
kwargs1 = dict(c='deepskyblue', label='True', zorder=10)

plt.style.use('seaborn-colorblind')
plt.style.use('seaborn-whitegrid')
fig, ax = plt.subplots(2, 4, figsize=(15,6))

ax[0][0].plot(ro,ro, **kwargs1)
ax[0][0].scatter(ro,predict_ro, **kwargs)
ax[0][1].plot(e,e, **kwargs1)
ax[0][1].scatter(e,predict_e, **kwargs)
ax[0][2].plot(vx,vx, **kwargs1)
ax[0][2].scatter(vx,predict_vx, **kwargs)
ax[0][3].plot(vy,vy, **kwargs1)
ax[0][3].scatter(vy,predict_vy, **kwargs)
ax[1][0].plot(vz,vz, **kwargs1)
ax[1][0].scatter(vz,predict_vz, **kwargs)
ax[1][1].plot(By,By, **kwargs1)
ax[1][1].scatter(By,predict_By, **kwargs)
ax[1][2].plot(Bz,Bz, **kwargs1)
ax[1][2].scatter(Bz,predict_Bz, **kwargs)

ax[0][0].set_xlabel("True")
ax[0][0].set_ylabel("Predict")
ax[0][1].set_xlabel("True")
ax[0][1].set_ylabel("Predict")
ax[0][2].set_xlabel("True")
ax[0][2].set_ylabel("Preidct")
ax[0][3].set_xlabel("True")
ax[0][3].set_ylabel("Predit")
ax[1][0].set_xlabel("True")
ax[1][0].set_ylabel("Predict")
ax[1][1].set_xlabel("True")
ax[1][1].set_ylabel("Predict")
ax[1][2].set_xlabel("True")
ax[1][2].set_ylabel("Preidct")

ax[0][0].set_title("fro "+str(R2[0].data))
ax[0][1].set_title("fe "+str(R2[1].data))
ax[0][2].set_title("fvx "+str(R2[2].data))
ax[0][3].set_title("fvy "+str(R2[3].data))
ax[1][0].set_title("fvz "+str(R2[4].data))
ax[1][1].set_title("fBy "+str(R2[5].data))
ax[1][2].set_title("fBz "+str(R2[6].data))

ax[0][0].legend()
ax[0][1].legend()
ax[0][2].legend()
ax[0][3].legend()
ax[1][0].legend()
ax[1][1].legend()
ax[1][2].legend()

ax[1][3].axis("off")

plt.tight_layout()
plt.savefig('./'+ DATA_SIZE + '/'+LOSS_FUNCTION+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+N_DATA+'/flux_R2_1.png')
plt.close()


"""

plt.plot(data_f[:,0],data_f[:,0])
plt.scatter(data_f[:,0], predict_1[:,0])
plt.show()


"""
plt.style.use('seaborn-colorblind')
plt.style.use('seaborn-whitegrid')
fig, ax = plt.subplots(2, 4, figsize=(15,6))
ax[0][0].plot(data_f[:,0],data_f[:,0], **kwargs1)
ax[0][0].scatter(data_f[:,0],predict_1[:,0], **kwargs)


ax[0][1].plot(data_f[:,1],data_f[:,1], **kwargs1)
ax[0][1].scatter(data_f[:,1],predict_1[:,1], **kwargs)


ax[0][2].plot(data_f[:,2],data_f[:,2], **kwargs1)
ax[0][2].scatter(data_f[:,2],predict_1[:,2], **kwargs)


ax[0][3].plot(data_f[:,3],data_f[:,3], **kwargs1)
ax[0][3].scatter(data_f[:,3],predict_1[:,3], **kwargs)


ax[1][0].plot(data_f[:,4],data_f[:,4], **kwargs1)
ax[1][0].scatter(data_f[:,4],predict_1[:,4], **kwargs)


ax[1][1].plot(data_f[:,5],data_f[:,5], **kwargs1)
ax[1][1].scatter(data_f[:,5],predict_1[:,5], **kwargs)


ax[1][2].plot(data_f[:,6],data_f[:,6], **kwargs1)
ax[1][2].scatter(data_f[:,6],predict_1[:,6], **kwargs)


ax[0][0].set_xlabel("True")
ax[0][0].set_ylabel("predict")
ax[0][1].set_xlabel("True")
ax[0][1].set_ylabel("Predict")
ax[0][2].set_xlabel("True")
ax[0][2].set_ylabel("Preidct")
ax[0][3].set_xlabel("True")
ax[0][3].set_ylabel("Predit")
ax[1][0].set_xlabel("True")
ax[1][0].set_ylabel("predict")
ax[1][1].set_xlabel("True")
ax[1][1].set_ylabel("Predict")
ax[1][2].set_xlabel("True")
ax[1][2].set_ylabel("Preidct")

ax[0][0].set_title("dfro "+str(R2_df[0].data))
ax[0][1].set_title("dfe" +str(R2_df[1].data))
ax[0][2].set_title("dfvx "+str(R2_df[2].data))
ax[0][3].set_title("dfvy "+str(R2_df[3].data))
ax[1][0].set_title("dfvz "+str(R2_df[4].data))
ax[1][1].set_title("dfBy "+str(R2_df[5].data))
ax[1][2].set_title("dfBz "+str(R2_df[6].data))

ax[0][0].legend()
ax[0][1].legend()
ax[0][2].legend()
ax[0][3].legend()
ax[1][0].legend()
ax[1][1].legend()
ax[1][2].legend()

ax[1][3].axis("off")

plt.tight_layout()
plt.savefig('./'+ DATA_SIZE+'/'+LOSS_FUNCTION+'/'+PREPROCESS+'/'+OPTIMIZER+'/'+N_DATA+'/flux_R2_d1.png')
plt.close()















