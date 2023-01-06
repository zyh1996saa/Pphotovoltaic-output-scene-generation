
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import tensorflow as tf



# In[]


    


# In[]


# In[]

pv_predict = np.zeros((16,96))
for i in range(16):
    tempdata = np.load('generated/data%s.npy'%i)
    pv_predict[i,:] = tempdata
    
pv_predict[:,:32] = 0
pv_predict[:,68:] = 0
# In[] 
sample_size = 16
data_shape = 96
# 采样标准正态分布
def sample_guass_noise(scale=1,size=(data_shape,)):
    return np.random.normal(loc=0,scale=1,size=size)

# 连乘alpha
def PI_alpha(alpha,t):
    temp = 1
    for i in range(t+1):
        temp *= alpha[i]
    return temp

pv_data_array = np.zeros((sample_size,data_shape))
for i in range(sample_size):
    start_point = random.randint(32,36)
    end_point = random.randint(68,72)
    max_point = int(start_point/2+end_point/2)
    max_value = random.uniform(0.4,2)
    value_series = np.array([(max_value*random.uniform(0.7,1)*(j-start_point)/(max_point-start_point)) if j<= max_point else (random.uniform(0.7,1)*(max_value-max_value*(j-max_point)/(end_point-max_point))) for j in range(start_point,end_point)])
    pv_data_array[i,start_point:end_point] = value_series
    
    
fig = plt.figure(figsize=(12,6), dpi=80)
#plt.figure(1)
ax1 = plt.subplot(211)
im1 = ax1.imshow(pv_data_array/pv_data_array.max())
ax1.set_title('real PV (pu)')

ax2 = plt.subplot(212)
im2 = ax2.imshow(pv_predict/pv_predict.max())
ax2.set_title('generated PV (pu)')

fig.colorbar(im1, ax=ax1, label='')
fig.colorbar(im2, ax=ax2, label='')
plt.show()



