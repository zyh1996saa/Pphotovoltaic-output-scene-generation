import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
# In[]

df = pd.read_excel('SolarForecast_20220105-20230105.xls')
pv_data = df.iloc[3:,5]

pv_data_array = np.zeros((int(pv_data.shape[0]/96),96))

for day in range(pv_data_array.shape[0]):
    pv_data_array[day,:] = pv_data[day*96:(day+1)*96]
    
del df, pv_data, day

# In[]
day = random.randint(0,pv_data_array.shape[0])
pv_mean = pv_data_array[:,48].mean()
pv_data_array = pv_data_array/pv_mean
plt.plot(pv_data_array[day])
del day

# In[]
sample_size = 1000000
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

# In[]


pv_data_array = np.zeros((sample_size,data_shape))
for i in range(sample_size):
    start_point = random.randint(32,36)
    end_point = random.randint(68,72)
    max_point = int(start_point/2+end_point/2)
    max_value = random.uniform(0.4,2)
    value_series = np.array([(max_value*random.uniform(0.7,1)*(j-start_point)/(max_point-start_point)) if j<= max_point else (random.uniform(0.7,1)*(max_value-max_value*(j-max_point)/(end_point-max_point))) for j in range(start_point,end_point)])
    pv_data_array[i,start_point:end_point] = value_series
# In[]


while 1:
    order = input('input something:')
    if order == 'quit':
        break
    else:
        random_int = random.randint(0,sample_size-1)
        print('random_int=%s'%random_int)
        plt.plot(pv_data_array[random_int,:])
        plt.show()
del order,random_int 
# In[]
repeat_time = 1

T = 1000
# 定义beta
beta_t = np.linspace(0.0001,0.02,T)

# 定义alpha = 1- beta
alpha_t = 1 - beta_t

# 定义alpha_ba_t = alpha_1 * alpha_2 * ... * aplha_t 
alpha_ba_t = np.array([PI_alpha(alpha_t,t) for t in range(T)])

# 定义x0的系数 = sqrt(alpha_ba)
x0_t_coeff = np.sqrt(alpha_ba_t)

# 定义噪声z0的系数 = sqrt(1-alpha_ba)
noise_t_coeff = np.sqrt(1-alpha_ba_t)

t_for_each_data = np.random.randint(low=0, high=T, size=sample_size*repeat_time)
# In[]
ori_data = pv_data_array
for i in range(repeat_time-1):
    ori_data = np.vstack((ori_data,pv_data_array))
# diffusion_tri
diff_tri = np.zeros((sample_size*repeat_time,data_shape))
noise_t = np.zeros((sample_size*repeat_time,data_shape))
for data_label in range(sample_size):
    for r in range(repeat_time):
        t = t_for_each_data[data_label+r*repeat_time]
        noise_t[data_label+r*repeat_time,:] = sample_guass_noise()
        diff_tri[data_label+r*repeat_time,:] = (x0_t_coeff[t]*ori_data[data_label+r*repeat_time]).reshape(1,data_shape) + (noise_t_coeff[t]*noise_t[data_label+r*repeat_time,:]).reshape(1,data_shape)
        
# In[]
while 1:
    order = input('input something:')
    if order == 'quit':
        break
    else:
        random_int = random.randint(0,sample_size*repeat_time-1)
        print('random_int=%s'%random_int)
        t = t_for_each_data[random_int]
        print('t=%s'%t)
        plt.plot(ori_data[random_int,:],color='b')
        plt.plot(diff_tri[random_int,:],color='r')
        plt.plot(noise_t[random_int,:],color='c')
        #print(x0_t_coeff[t],)
        plt.show()
del order,random_int

# In[]

# 定义神经网络

input0 = tf.keras.Input(shape=(data_shape,))
input1 = tf.keras.Input(shape=(1,))
x1 = tf.keras.layers.Dense(128,name='dense1',activation='gelu')(input1)
x1 = tf.keras.layers.Dense(128,name='dense2',activation='gelu')(x1)
x1 = tf.keras.layers.Dense(96,name='dense3',)(x1)
x0 = tf.keras.layers.Dense(256,name='dense4',activation='gelu',)(input0)
#x0 = tf.keras.layers.Dense(128,name='dense5',activation='gelu',)(x0)
x0 = tf.keras.layers.Dense(96,name='dense6',)(x0)
x = tf.keras.layers.Concatenate(axis=1)([x0, x1])
x = tf.keras.layers.Dense(256,name='dense7',activation='gelu',)(x)
x = tf.keras.layers.Dense(128,name='dense8',activation='gelu',)(x)
output = tf.keras.layers.Dense(data_shape,name='dense9',)(x)
model = tf.keras.Model([input0,input1], output)
# In[]

model_input0 = diff_tri
model_input1 = t_for_each_data/T
model_output = noise_t
model.compile(optimizer='Adam',loss='MSE')
model.fit([model_input0,model_input1],noise_t,epochs=10000,batch_size=2048)
# In[]
'''
diff_tri[data_label,:] = (x0_t_coeff[t]*ori_data[data_label]).reshape(1,3) + (noise_t_coeff[t]*noise_t[data_label,:]).reshape(1,3)
'''
while 1:
    order = input('input something:')
    if order == 'quit':
        break
    else:
        random_int = random_int = random.randint(0,sample_size-1)
        x0_true = ori_data[random_int]
        t = t_for_each_data[random_int]
        xt = (x0_t_coeff[t]*ori_data[random_int]).reshape(1,data_shape) + (noise_t_coeff[t]*noise_t[random_int,:]).reshape(1,data_shape)
        xt_read = diff_tri[random_int,:]
        
        
        noise_predict = model.predict([xt.reshape(1,data_shape),np.array([[t/1000]])])
        noise_true = noise_t[random_int,:]
        
        x0_predict = (xt-noise_t_coeff[t]*noise_predict.reshape(data_shape))/x0_t_coeff[t]
        x0_true_cal = (xt-noise_t_coeff[t]*noise_true.reshape(data_shape))/x0_t_coeff[t]
        
        plt.plot(x0_predict.reshape(data_shape),color='red')    
        plt.plot(x0_true_cal.reshape(data_shape),color='pink')
        plt.plot(noise_predict.reshape(data_shape),color='c')
        plt.plot(noise_true.reshape(data_shape),color='b')
        print('random_int',random_int)
        plt.show()
# In[]

#model.save('model')
# In[]
loaded_model = tf.keras.models.load_model('model')
# In[]
while 1:
    order = input('input something:')
    if order == 'quit':
        break
    else:
        for t in range(T,0,-1):
            if t == T:
                xt = sample_guass_noise()
            else:
                xt = xt_red_1
            noise_t_pre = loaded_model.predict([xt.reshape(1,data_shape),np.array([[t/1000]])],verbose = 0).reshape(data_shape)
            alpha = alpha_t[t-1]
            alpha_ba = alpha_ba_t[t-1]
            z = sample_guass_noise()
            xt_red_1 = (1/np.sqrt(alpha))*(xt-(1-alpha)*noise_t_pre/np.sqrt(1-alpha_ba))+z*np.sqrt(1-alpha)
            print('\r t=%s'%(t-1),end='\r')
        plt.plot(xt)
        plt.show()
# In[]
for i in range(16):
    for t in range(T,0,-1):
        if t == T:
            xt = sample_guass_noise()
        else:
            xt = xt_red_1
        noise_t_pre = loaded_model.predict([xt.reshape(1,data_shape),np.array([[t/1000]])],verbose = 0).reshape(data_shape)
        alpha = alpha_t[t-1]
        alpha_ba = alpha_ba_t[t-1]
        z = sample_guass_noise()
        xt_red_1 = (1/np.sqrt(alpha))*(xt-(1-alpha)*noise_t_pre/np.sqrt(1-alpha_ba))+z*np.sqrt(1-alpha)
        print('\r t=%s,i=%s'%(t-1,i),end='\r')
    np.save('generated/data%s'%i,xt)