#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import random
import copy
import math
from collections import OrderedDict, Counter
#from multiprocessing import Pool
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
#from scipy.integrate import odeint
from scipy.integrate import solve_ivp


# In[2]:


_Data_PATH_XX_ = './simulation_results/'
_Figure_PATH_XX_ = './figures/'


# In[3]:


N = 2000        #节点数量

mu = 1          #恢复速率
beta = 0.6      #传播速率与接触人数的乘积
alpha = 0.5     #防护效果
w = 1         #行为更新速率 （如果结果不稳定，可能是由于更新过快的原因）
c0 = 1          #采取防护措施的花费
cI = 6          #被感染后的代价
ita = 0.001          #社会一致性的影响强度
k = 10           #每时间步平均接触人数或者网络度
gamma = 1       #由于媒体宣传，夸大感染能力或者对感染能力的误判

T = 1000         #运行时间
repeat_num = 30  #运行次数

#Init_condition_list = [0.01, 0.05]
n_inf = 0.01
n_take = 0.05    #初始采取防护措施的比例

beta_list = [0.02 * i for i in range(51)]


# In[4]:


filename1 = 'wellmixinf_N'+str(N)+'k' + str(k) + '_mu'+ str(mu) + '_alpha' + str(alpha) + '_w' + str(w) + '_cI' +str(cI) +'_ita'+str(ita)+'gamma'+str(gamma)+'n_inf'+str(n_inf)+'n_take'+str(n_take)
filename2 = 'wellmixtake_N'+str(N)+'k' + str(k) + '_mu'+ str(mu) + '_alpha' + str(alpha) + '_w' + str(w) + '_cI' +str(cI) +'_ita'+str(ita)+'gamma'+str(gamma)+'n_inf'+str(n_inf)+'n_take'+str(n_take)


# In[5]:


#well-mixed population and 节点状态初始化

# 节点初始状态
def initial_state(N,n_inf,n_take):
    sus_set = set([i for i in range(N)])
    inf_set = set()
    
    
    i = 0
    while i < N * n_inf:
        j = np.random.randint(N)
        while j in inf_set:
            j = np.random.randint(N)
        
        inf_set.add(j)
        sus_set.remove(j)
        i = i + 1
    
    inf_non = inf_set.copy()
    inf_take = set()      # 这个是指潜在的采取防护措施的个体，一旦恢复，则延续之前的防护举措。
        
    
    sus_non = sus_set.copy()
    sus_take = set()
    
    i = 0
    while i < len(sus_set) * n_take:
        j = np.random.choice(list(sus_non))
        sus_non.remove(j)
        sus_take.add(j)
        
        i = i + 1
            
    return inf_set, inf_non, inf_take, sus_set, sus_non, sus_take


# In[6]:


#定义无防护时的传播过程
def spread_without_precaution(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take):
    i = random.choice(list(sus_non))
    inf_set.add(i)
    inf_non.add(i)
    sus_set.remove(i)
    sus_non.remove(i)
    
    return inf_set, inf_non, inf_take, sus_set, sus_non, sus_take

#定义有防护时的传播过程
def spread_precaution(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take):
    i = random.choice(list(sus_take))
    inf_set.add(i)
    inf_take.add(i)
    sus_set.remove(i)
    sus_take.remove(i)
    
    return inf_set, inf_non, inf_take, sus_set, sus_non, sus_take

#定义疾病恢复过程
def recover(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take):
    i = random.choice(list(inf_set))
    inf_set.remove(i)
    sus_set.add(i)
    if i in inf_non:
        inf_non.remove(i)
        sus_non.add(i)
    else:
        inf_take.remove(i)
        sus_take.add(i)
    
    return inf_set, inf_non, inf_take, sus_set, sus_non, sus_take

#定义策略转换过程(不防护到防护)
def behavior_non_to_take(sus_non, sus_take):
    i = random.choice(list(sus_non))
    sus_non.remove(i)
    sus_take.add(i)
    
    return sus_non, sus_take

#定义策略转换过程(防护到不防护)
def behavior_take_to_non(sus_non, sus_take):
    i = random.choice(list(sus_take))
    sus_take.remove(i)
    sus_non.add(i)
    
    return sus_non, sus_take


# In[7]:


#定义各事件的速率
def event_rate(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, N, mu, beta, alpha, w, c0, cI, ita, k, gamma):
    rate_spread_without_precaution = beta * k * len(sus_non) * len(inf_set)/N
    rate_spread_precaution = (1-alpha) * beta * k * len(sus_take) * len(inf_set)/N
    rate_recovery = mu * len(inf_set)
    
    
    I = len(inf_set)/N
    p = len(sus_take)/len(sus_set)
    tem = w * (-c0 + cI*(np.exp(-(1-alpha)*beta*k*I*gamma) - np.exp(-beta*k*I*gamma)) + 2*ita*(2*p-1))
    if tem > 0:
        rate_non_to_take = tem * len(sus_non) * len(sus_take)/(N-len(inf_set))
        rate_take_to_non = 0
    else:
        rate_non_to_take = 0
        rate_take_to_non = -tem * len(sus_take) * len(sus_non)/(N-len(inf_set))
    
    return rate_spread_without_precaution, rate_spread_precaution, rate_recovery, rate_non_to_take, rate_take_to_non


# In[8]:


def Gillespie_simulation(N, n_inf, n_take, mu, beta, alpha, w, c0, cI, ita, k, gamma, T):
    
    inf_set, inf_non, inf_take, sus_set, sus_non, sus_take = initial_state(N,n_inf,n_take)
    t = 0
    inf_n = 0
    b_take_n = 0
    i_count = 0
    while t < T:
        r1 = random.random()
        r2 = random.random()
        
        rate_spread_without_precaution, rate_spread_precaution, rate_recovery, rate_non_to_take, rate_take_to_non = event_rate(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, N, mu, beta, alpha, w, c0, cI, ita, k, gamma)
        rate_sum = rate_spread_without_precaution + rate_spread_precaution + rate_recovery + rate_non_to_take + rate_take_to_non
        
        
        if rate_sum == 0:
            inf_ave = len(inf_set) / N
            b_take_ave = len(sus_take) / len(sus_set)
            break
        
        tau = -math.log(r1) / rate_sum
        
        if r2 * rate_sum <  rate_spread_without_precaution:
            inf_set, inf_non, inf_take, sus_set, sus_non, sus_take = spread_without_precaution(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take)
        elif r2 * rate_sum < rate_spread_without_precaution + rate_spread_precaution:
            inf_set, inf_non, inf_take, sus_set, sus_non, sus_take = spread_precaution(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take)
        elif r2 * rate_sum < rate_spread_without_precaution + rate_spread_precaution + rate_recovery:
            inf_set, inf_non, inf_take, sus_set, sus_non, sus_take = recover(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take)
        else:
            if rate_non_to_take == 0:
                sus_non, sus_take = behavior_take_to_non(sus_non, sus_take)
            else:
                sus_non, sus_take = behavior_non_to_take(sus_non, sus_take)
        
        t = t + tau
        

        # 扰动系统：在有限人群中当采取防护措施的人比例归0，进行系统扰动（拟稳态）
        if t > T-10:
            inf_n = inf_n + len(inf_set) / N
            b_take_n = b_take_n + len(sus_take) / len(sus_set)
            i_count = i_count + 1
        
    if i_count != 0:
        inf_ave = inf_n / i_count
        b_take_ave = b_take_n / i_count
    return inf_ave, b_take_ave    


# In[9]:


# 定义多次仿真的动力学过程
def repeated_simulation(N, n_inf, n_take, mu, beta, alpha, w, c0, cI, ita, k, gamma, T, repeat_num):
    infected_ratio = []
    takemeasure_ratio = []
    for i in range(repeat_num):
        inf_ave, take_ave = Gillespie_simulation(N, n_inf, n_take, mu, beta, alpha, w, c0, cI, ita, k, gamma, T)
        infected_ratio.append(inf_ave)
        takemeasure_ratio.append(take_ave)
    
    return infected_ratio, takemeasure_ratio


# In[10]:


#运行仿真
rho_I_dict = {}
rho_take_dict = {}
for beta in beta_list:
    infected_ratio, takemeasure_ratio = repeated_simulation(N, n_inf, n_take, mu, beta, alpha, w, c0, cI, ita, k, gamma, T, repeat_num)
    rho_I_dict[beta] = infected_ratio
    rho_take_dict[beta] = takemeasure_ratio


# In[ ]:


#结果存储
np.save(_Data_PATH_XX_ + filename1 + '.npy', rho_I_dict)
np.save(_Data_PATH_XX_ + filename2 + '.npy', rho_take_dict)

