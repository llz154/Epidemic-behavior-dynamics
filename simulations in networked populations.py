#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import random
import copy
import math
from collections import OrderedDict, Counter
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from multiprocessing import Pool   #python环境
#from multiprocess import Pool       #jupyter环境


# In[ ]:


_Data_PATH_XX_ = './simulation_results/'
_Figure_PATH_XX_ = './figures/'


# In[ ]:



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

T = 2000         #运行时间
repeat_num = 30  #运行次数

#Init_condition_list = [0.01, 0.2]
n_inf = 0.01
n_take = 0.05    #初始采取防护措施的比例

beta_list = [0.02 * i for i in range(51)]
#beta_list = [0.4,0.5]
network_index = 0
network_list = ['regular','random','powerlaw']


# In[ ]:


filename1 = network_list[network_index] + 'inf_N'+str(N)+'k' + str(k) + '_mu'+ str(mu) + '_alpha' + str(alpha) + '_w' + str(w) + '_cI' +str(cI) +'_ita'+str(ita)+'gamma'+str(gamma)+'n_inf'+str(n_inf)+'n_take'+str(n_take)
filename2 = network_list[network_index] + 'take_N'+str(N)+'k' + str(k) + '_mu'+ str(mu) + '_alpha' + str(alpha) + '_w' + str(w) + '_cI' +str(cI) +'_ita'+str(ita)+'gamma'+str(gamma)+'n_inf'+str(n_inf)+'n_take'+str(n_take)


# In[ ]:


#生成网络（节点列表、连边列表）可存储为字典
def graph_generating(N, k, network_index):
    if network_index == 0:
        G = nx.random_regular_graph(d=k, n=N)
    if network_index == 1:
        G = nx.gnm_random_graph(N, N*k/2)
    if network_index == 2:
        G = nx.barabasi_albert_graph(n=N, m=int(k/2))
    
    nodes_list = list(G.nodes())
    edges_list = list(G.edges())
    
    neighbor_dict = {}   #存储与每个节点的邻居（type: list）
    for i in nodes_list:
        neighbor_dict[i] = list(G.neighbors(i))
    
    #print([len(nodes_list), len(edges_list)])
    
    return nodes_list, edges_list, neighbor_dict


# In[ ]:


#计算一个未采取防护措施的S节点被感染的速率
def calculate_rate_without_precaution(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, neighbor_dict, beta, alpha, w, c0, cI, ita, gamma, node):
    
    if node in sus_non:
        if len(neighbor_dict[node]) == 0:
            rate_spread = 0
            rate_behavior = 0
        else:
            tem = 0
            sus_local = set()
            sus_take_local = set()
            for j in neighbor_dict[node]:
                if j in inf_set:
                    tem = tem + 1
                else:
                    sus_local.add(j)
                    if j in sus_take:
                        sus_take_local.add(j)
            
            rate_spread = beta * tem
            
            if len(sus_local) == 0:
                rate_behavior = 0
            else:
                p = len(sus_take_local) / len(sus_local)
                #随机选择一个S状态节点观察
                node_s = random.choice(list(sus_local))
                if node_s in sus_take_local:
                    payoff_i = -cI * (1 - np.exp(-beta * tem * gamma)) + ita * (1 - 2 * p)
                    tem_s_inf = 0
                    tem_s_sus = 0
                    tem_s_sus_take = 0
                    for neigh_s in neighbor_dict[node_s]:
                        if neigh_s in inf_set:
                            tem_s_inf = tem_s_inf + 1
                        else:
                            tem_s_sus = tem_s_sus + 1
                            if neigh_s in sus_take:
                                tem_s_sus_take = tem_s_sus_take + 1
                    
                    p_s = tem_s_sus_take / tem_s_sus
                    
                    payoff_node_s = -c0 - cI * (1 - np.exp(-(1-alpha) * beta * tem_s_inf * gamma)) + ita * (2 * p_s - 1)
                    rate_behavior = max(0, w * (payoff_node_s - payoff_i))
                else:
                    rate_behavior = 0
    else:
        rate_spread = 0
        rate_behavior = 0

    return rate_spread, rate_behavior

#计算一个采取防护措施的S节点被感染的速率
def calculate_rate_with_precaution(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, neighbor_dict, beta, alpha, w, c0, cI, ita, gamma, node):
    rate_spread = 0
    rate_behavior = 0
    
    if node in sus_take:
        if len(neighbor_dict[node]) == 0:
            rate_spread = 0
            rate_behavior = 0
        else:
            tem = 0
            sus_local = set()
            sus_take_local = set()
            for j in neighbor_dict[node]:
                if j in inf_set:
                    tem = tem + 1
                else:
                    sus_local.add(j)
                    if j in sus_take:
                        sus_take_local.add(j)
            
            rate_spread = (1 - alpha) * beta * tem
            
            if len(sus_local) == 0:
                rate_behavior = 0
            else:
                p = len(sus_take_local) / len(sus_local)
                #随机选择一个S状态节点观察
                node_s = random.choice(list(sus_local))
                if node_s in sus_take_local:
                    rate_behavior = 0
                else:
                    payoff_i = -c0 - cI * (1 - np.exp(-(1-alpha)*beta * tem * gamma)) + ita * (2 * p - 1)
                    
                    tem_s_inf = 0
                    tem_s_sus = 0
                    tem_s_sus_take = 0
                    for neigh_s in neighbor_dict[node_s]:
                        if neigh_s in inf_set:
                            tem_s_inf = tem_s_inf + 1
                        else:
                            tem_s_sus = tem_s_sus + 1
                            if neigh_s in sus_take:
                                tem_s_sus_take = tem_s_sus_take + 1
                    
                    
                    p_s = tem_s_sus_take / tem_s_sus
                    payoff_node_s = - cI * (1 - np.exp(-beta * tem_s_inf * gamma)) + ita * (1 - 2 * p_s)
                    rate_behavior = max(0, w * (payoff_node_s - payoff_i))
    else:
        rate_spread = 0
        rate_behavior = 0
    
    return rate_spread, rate_behavior


# In[ ]:


#节点状态初始化与各事件类型总速率
def initial_state(nodes_list, n_inf, n_take, neighbor_dict, beta, alpha, w, c0, cI, ita, gamma):
    sus_set = set([i for i in nodes_list])
    inf_set = set()
    N = len(nodes_list)
    
    
    i = 0
    while i < N * n_inf:
        j = np.random.choice(list(sus_set))
        
        sus_set.remove(j)
        inf_set.add(j)
        
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
        
    # 记录每个节点发生感染、防护策略转变的速率
    rate_spread_without_precaution = [0 for i in nodes_list]
    rate_spread_with_precaution = [0 for i in nodes_list]
    rate_non_to_take = [0 for i in nodes_list]
    rate_take_to_non = [0 for i in nodes_list]
    
    for i in sus_non:
        rate_spread, rate_behavior = calculate_rate_without_precaution(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, neighbor_dict, beta, alpha, w, c0, cI, ita, gamma, i)
        
        rate_spread_without_precaution[i] = rate_spread
        rate_non_to_take[i] = rate_behavior
    
    for i in sus_take:
        rate_spread, rate_behavior = calculate_rate_with_precaution(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, neighbor_dict, beta, alpha, w, c0, cI, ita, gamma, i)
        
        rate_spread_with_precaution[i] = rate_spread
        rate_take_to_non[i] = rate_behavior

    return inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, rate_spread_without_precaution, rate_spread_with_precaution, rate_non_to_take, rate_take_to_non


# In[ ]:


#定义可能发生的事件
# 无防护措施节点的感染

def spreading_without_precaution(nodes_list, neighbor_dict, inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, rate_spread_without_precaution, rate_spread_with_precaution, rate_non_to_take, rate_take_to_non):
    #按照各节点传播权重选择
    #node_selected = random.choices(nodes_list, weights=rate_spread_without_precaution, k=1)[0]
    
    temp_list = []
    weigh_list = []
    for i in sus_non:
        if rate_spread_without_precaution[i] != 0:
            temp_list.append(i)
            weigh_list.append(rate_spread_without_precaution[i])
    
    node_selected = random.choices(temp_list, weights=weigh_list, k=1)[0]
    
    inf_set.add(node_selected)
    inf_non.add(node_selected)
    sus_set.remove(node_selected)
    sus_non.remove(node_selected)
    
    #更新当前节点的速率权重
    rate_spread_without_precaution[node_selected] = 0
    rate_non_to_take[node_selected] = 0
    
    #更新邻居节点的速率权重及二阶邻居的速率权重
    if len(neighbor_dict[node_selected]) != 0:
        for i in neighbor_dict[node_selected]:
            if i in sus_non:
                rate_spread, rate_behavior = calculate_rate_without_precaution(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, neighbor_dict, beta, alpha, w, c0, cI, ita, gamma, i)
                rate_spread_without_precaution[i] = rate_spread
                rate_non_to_take[i] = rate_behavior
            if i in sus_take:
                rate_spread, rate_behavior = calculate_rate_with_precaution(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, neighbor_dict, beta, alpha, w, c0, cI, ita, gamma, i)
                rate_spread_with_precaution[i] = rate_spread
                rate_take_to_non[i] = rate_behavior
            
            #更新二阶邻居速率权重
            for j in neighbor_dict[i]:
                if j in sus_non:
                    rate_spread, rate_behavior = calculate_rate_without_precaution(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, neighbor_dict, beta, alpha, w, c0, cI, ita, gamma, j)
                    rate_spread_without_precaution[j] = rate_spread
                    rate_non_to_take[j] = rate_behavior
                if j in sus_take:
                    rate_spread, rate_behavior = calculate_rate_with_precaution(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, neighbor_dict, beta, alpha, w, c0, cI, ita, gamma, j)
                    rate_spread_with_precaution[j] = rate_spread
                    rate_take_to_non[j] = rate_behavior

    return inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, rate_spread_without_precaution, rate_spread_with_precaution, rate_non_to_take, rate_take_to_non

#有防护措施节点的感染
def spreading_with_precaution(nodes_list, neighbor_dict, inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, rate_spread_without_precaution, rate_spread_with_precaution, rate_non_to_take, rate_take_to_non):
    #按照各节点传播权重选择
    #node_selected = random.choices(nodes_list, weights=rate_spread_with_precaution, k=1)[0]
    
    temp_list = []
    weigh_list = []
    
    for i in sus_take:
        if rate_spread_with_precaution[i] != 0:
            temp_list.append(i)
            weigh_list.append(rate_spread_with_precaution[i])
    
    node_selected = random.choices(temp_list, weights=weigh_list, k=1)[0]
    
    inf_set.add(node_selected)
    inf_take.add(node_selected)
    sus_set.remove(node_selected)
    sus_take.remove(node_selected)
    
    #更新速率权重
    rate_spread_with_precaution[node_selected] = 0
    rate_take_to_non[node_selected] = 0
    
    #更新邻居节点的速率权重及二阶邻居的速率权重
    if len(neighbor_dict[node_selected]) != 0:
        for i in neighbor_dict[node_selected]:
            if i in sus_non:
                rate_spread, rate_behavior = calculate_rate_without_precaution(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, neighbor_dict, beta, alpha, w, c0, cI, ita, gamma, i)
                rate_spread_without_precaution[i] = rate_spread
                rate_non_to_take[i] = rate_behavior
            if i in sus_take:
                rate_spread, rate_behavior = calculate_rate_with_precaution(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, neighbor_dict, beta, alpha, w, c0, cI, ita, gamma, i)
                rate_spread_with_precaution[i] = rate_spread
                rate_take_to_non[i] = rate_behavior
            
            #更新二阶邻居速率权重
            for j in neighbor_dict[i]:
                if j in sus_non:
                    rate_spread, rate_behavior = calculate_rate_without_precaution(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, neighbor_dict, beta, alpha, w, c0, cI, ita, gamma, j)
                    rate_spread_without_precaution[j] = rate_spread
                    rate_non_to_take[j] = rate_behavior
                if j in sus_take:
                    rate_spread, rate_behavior = calculate_rate_with_precaution(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, neighbor_dict, beta, alpha, w, c0, cI, ita, gamma, j)
                    rate_spread_with_precaution[j] = rate_spread
                    rate_take_to_non[j] = rate_behavior
    
    return inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, rate_spread_without_precaution, rate_spread_with_precaution, rate_non_to_take, rate_take_to_non

#节点恢复
def recover(nodes_list, neighbor_dict, inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, rate_spread_without_precaution, rate_spread_with_precaution, rate_non_to_take, rate_take_to_non):
    
    node_selected = random.choice(list(inf_set))
    
    if node_selected in inf_non:
        inf_set.remove(node_selected)
        inf_non.remove(node_selected)
        sus_set.add(node_selected)
        sus_non.add(node_selected)
        
        #更新自身的速率权重
        r1, r2 = calculate_rate_without_precaution(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, neighbor_dict, beta, alpha, w, c0, cI, ita, gamma, node_selected)
        rate_spread_without_precaution[node_selected] = r1
        rate_non_to_take[node_selected] = r2
            
    else:
        inf_set.remove(node_selected)
        inf_take.remove(node_selected)
        sus_set.add(node_selected)
        sus_take.add(node_selected)
        
        #更新自身的速率权重
        r1, r2 = calculate_rate_with_precaution(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, neighbor_dict, beta, alpha, w, c0, cI, ita, gamma, node_selected)
        rate_spread_with_precaution[node_selected] = r1
        rate_take_to_non[node_selected] = r2
    
    #更新邻居节点的速率权重及二阶邻居的速率权重
    if len(neighbor_dict[node_selected]) != 0:
        for i in neighbor_dict[node_selected]:
            if i in sus_non:
                rate_spread, rate_behavior = calculate_rate_without_precaution(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, neighbor_dict, beta, alpha, w, c0, cI, ita, gamma, i)
                rate_spread_without_precaution[i] = rate_spread
                rate_non_to_take[i] = rate_behavior
            if i in sus_take:
                rate_spread, rate_behavior = calculate_rate_with_precaution(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, neighbor_dict, beta, alpha, w, c0, cI, ita, gamma, i)
                rate_spread_with_precaution[i] = rate_spread
                rate_take_to_non[i] = rate_behavior
            
            #更新二阶邻居速率权重
            for j in neighbor_dict[i]:
                if j in sus_non:
                    rate_spread, rate_behavior = calculate_rate_without_precaution(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, neighbor_dict, beta, alpha, w, c0, cI, ita, gamma,j)
                    rate_spread_without_precaution[j] = rate_spread
                    rate_non_to_take[j] = rate_behavior
                if j in sus_take:
                    rate_spread, rate_behavior = calculate_rate_with_precaution(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, neighbor_dict, beta, alpha, w, c0, cI, ita, gamma, j)
                    rate_spread_with_precaution[j] = rate_spread
                    rate_take_to_non[j] = rate_behavior
    
    return inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, rate_spread_without_precaution, rate_spread_with_precaution, rate_non_to_take, rate_take_to_non

#定义行为策略的转移：non-take
def behavior_non_to_take(nodes_list, neighbor_dict, inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, rate_spread_without_precaution, rate_spread_with_precaution, rate_non_to_take, rate_take_to_non):
    
    #按照各节点转移权重选择
    #node_selected = random.choices(nodes_list, weights=rate_non_to_take, k=1)[0]
    
    temp_list = []
    weigh_list = []
    
    for i in sus_non:
        if rate_non_to_take[i] != 0:
            temp_list.append(i)
            weigh_list.append(rate_non_to_take[i])
    
    node_selected = random.choices(temp_list, weights=weigh_list, k=1)[0]
    
    
    sus_non.remove(node_selected)
    sus_take.add(node_selected)
    
    #自身节点权重速率更新
    rate_spread_without_precaution[node_selected] = 0
    rate_non_to_take[node_selected] = 0
    r1, r2 = calculate_rate_with_precaution(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, neighbor_dict, beta, alpha, w, c0, cI, ita, gamma, node_selected)
    rate_spread_with_precaution[node_selected] = r1
    rate_take_to_non[node_selected] = r2
    
    #更新邻居节点的速率权重及二阶邻居的速率权重
    if len(neighbor_dict[node_selected]) != 0:
        for i in neighbor_dict[node_selected]:
            if i in sus_non:
                rate_spread, rate_behavior = calculate_rate_without_precaution(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, neighbor_dict, beta, alpha, w, c0, cI, ita, gamma, i)
                rate_spread_without_precaution[i] = rate_spread
                rate_non_to_take[i] = rate_behavior
            if i in sus_take:
                rate_spread, rate_behavior = calculate_rate_with_precaution(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, neighbor_dict, beta, alpha, w, c0, cI, ita, gamma, i)
                rate_spread_with_precaution[i] = rate_spread
                rate_take_to_non[i] = rate_behavior
            
            #更新二阶邻居速率权重
            for j in neighbor_dict[i]:
                if j in sus_non:
                    rate_spread, rate_behavior = calculate_rate_without_precaution(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, neighbor_dict, beta, alpha, w, c0, cI, ita, gamma, j)
                    rate_spread_without_precaution[j] = rate_spread
                    rate_non_to_take[j] = rate_behavior
                if j in sus_take:
                    rate_spread, rate_behavior = calculate_rate_with_precaution(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, neighbor_dict, beta, alpha, w, c0, cI, ita, gamma, j)
                    rate_spread_with_precaution[j] = rate_spread
                    rate_take_to_non[j] = rate_behavior
    
    return inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, rate_spread_without_precaution, rate_spread_with_precaution, rate_non_to_take, rate_take_to_non


#定义行为策略的转移：take-non
def behavior_take_to_non(nodes_list, neighbor_dict, inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, rate_spread_without_precaution, rate_spread_with_precaution, rate_non_to_take, rate_take_to_non):
    
    #node_selected = random.choices(nodes_list, weights=rate_take_to_non, k=1)[0]
    
    temp_list = []
    weigh_list = []
    
    for i in sus_take:
        if rate_take_to_non[i] != 0:
            temp_list.append(i)
            weigh_list.append(rate_take_to_non[i])
    
    node_selected = random.choices(temp_list, weights=weigh_list, k=1)[0]

    
    sus_take.remove(node_selected)
    sus_non.add(node_selected)
    
    #自身节点权重速率更新
    rate_spread_with_precaution[node_selected] = 0
    rate_take_to_non[node_selected] = 0
    r1, r2 = calculate_rate_without_precaution(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, neighbor_dict, beta, alpha, w, c0, cI, ita, gamma, node_selected)
    rate_spread_without_precaution[node_selected] = r1
    rate_non_to_take[node_selected] = r2
    
    #更新邻居节点的速率权重及二阶邻居的速率权重
    if len(neighbor_dict[node_selected]) != 0:
        for i in neighbor_dict[node_selected]:
            if i in sus_non:
                rate_spread, rate_behavior = calculate_rate_without_precaution(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, neighbor_dict, beta, alpha, w, c0, cI, ita, gamma, i)
                rate_spread_without_precaution[i] = rate_spread
                rate_non_to_take[i] = rate_behavior
            if i in sus_take:
                rate_spread, rate_behavior = calculate_rate_with_precaution(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, neighbor_dict, beta, alpha, w, c0, cI, ita, gamma, i)
                rate_spread_with_precaution[i] = rate_spread
                rate_take_to_non[i] = rate_behavior
            
            #更新二阶邻居速率权重
            for j in neighbor_dict[i]:
                if j in sus_non:
                    rate_spread, rate_behavior = calculate_rate_without_precaution(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, neighbor_dict, beta, alpha, w, c0, cI, ita, gamma, j)
                    rate_spread_without_precaution[j] = rate_spread
                    rate_non_to_take[j] = rate_behavior
                if j in sus_take:
                    rate_spread, rate_behavior = calculate_rate_with_precaution(inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, neighbor_dict, beta, alpha, w, c0, cI, ita, gamma, j)
                    rate_spread_with_precaution[j] = rate_spread
                    rate_take_to_non[j] = rate_behavior
    
    return inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, rate_spread_without_precaution, rate_spread_with_precaution, rate_non_to_take, rate_take_to_non


# In[ ]:


#定义Gillespie仿真过程
def Gillespie_network(args):
    
    it_num, nodes_list, neighbor_dict, n_inf, n_take, beta, alpha, w, c0, cI, ita, gamma, mu, T = args
    inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, rate_spread_without_precaution, rate_spread_with_precaution, rate_non_to_take, rate_take_to_non = initial_state(nodes_list, n_inf, n_take, neighbor_dict, beta, alpha, w, c0, cI, ita, gamma)
    
    t = 0
    inf_n = 0
    b_take_n = 0
    i_count = 0
    
    while t < T:
        r1 = random.random()
        r2 = random.random()
        
        total_spread_without_p = sum(rate_spread_without_precaution)
        total_spread_with_p = sum(rate_spread_with_precaution)
        total_non_to_take = sum(rate_non_to_take)
        total_take_to_non = sum(rate_take_to_non)
        
        total_rate =  total_spread_without_p + total_spread_with_p + total_non_to_take + total_take_to_non + mu * len(inf_set)
        
        if total_rate == 0:
            inf_ratio = len(inf_set) / len(nodes_list)
            behavior_ratio = len(sus_take) / len(sus_set)
            break
        
        tau = -math.log(r1) / total_rate
        
        if r2 * total_rate < total_spread_without_p:
            inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, rate_spread_without_precaution, rate_spread_with_precaution, rate_non_to_take, rate_take_to_non = spreading_without_precaution(nodes_list, neighbor_dict, inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, rate_spread_without_precaution, rate_spread_with_precaution, rate_non_to_take, rate_take_to_non)
        elif r2 * total_rate < total_spread_without_p + total_spread_with_p:
            inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, rate_spread_without_precaution, rate_spread_with_precaution, rate_non_to_take, rate_take_to_non = spreading_with_precaution(nodes_list, neighbor_dict, inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, rate_spread_without_precaution, rate_spread_with_precaution, rate_non_to_take, rate_take_to_non)
        elif r2 * total_rate < total_spread_without_p + total_spread_with_p + total_non_to_take:
            inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, rate_spread_without_precaution, rate_spread_with_precaution, rate_non_to_take, rate_take_to_non = behavior_non_to_take(nodes_list, neighbor_dict, inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, rate_spread_without_precaution, rate_spread_with_precaution, rate_non_to_take, rate_take_to_non)
        elif r2 * total_rate < total_spread_without_p + total_spread_with_p + total_non_to_take + total_take_to_non:
            inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, rate_spread_without_precaution, rate_spread_with_precaution, rate_non_to_take, rate_take_to_non = behavior_take_to_non(nodes_list, neighbor_dict, inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, rate_spread_without_precaution, rate_spread_with_precaution, rate_non_to_take, rate_take_to_non)
        else:
            inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, rate_spread_without_precaution, rate_spread_with_precaution, rate_non_to_take, rate_take_to_non = recover(nodes_list, neighbor_dict, inf_set, inf_non, inf_take, sus_set, sus_non, sus_take, rate_spread_without_precaution, rate_spread_with_precaution, rate_non_to_take, rate_take_to_non)
        
        t = t + tau
        
        if t > T - 10:
            inf_n = inf_n + len(inf_set) / len(nodes_list)
            b_take_n = b_take_n + len(sus_take) / len(sus_set)
            i_count = i_count + 1
    
    
    if i_count != 0:
        inf_ratio = inf_n / i_count
        behavior_ratio = b_take_n / i_count
    
    return inf_ratio, behavior_ratio


# In[ ]:


"""#定义多次仿真取平均
def repeated_simulation(N, n_inf, n_take, mu, beta, alpha, w, c0, cI, ita, k, gamma, T, repeat_num, network_index):
    
    #生成网络
    nodes_list, edges_list, neighbor_dict = graph_generating(N, k, network_index)
    
    infected_ratio = []
    takemeasure_ratio = []
    
    for i in range(repeat_num):
        inf_ratio, behavior_ratio = Gillespie_network(nodes_list, neighbor_dict, n_inf, n_take, beta, alpha, w, c0, cI, ita, gamma, mu, T)
        infected_ratio.append(inf_ratio)
        takemeasure_ratio.append(behavior_ratio)
    
    return infected_ratio, takemeasure_ratio"""


# In[ ]:


if __name__ == "__main__":
    #运行仿真
    rho_I_dict = {}
    rho_take_dict = {}
    for beta in beta_list:
        #生成网络
        nodes_list, edges_list, neighbor_dict = graph_generating(N, k, network_index)
        infected_ratio = []
        takemeasure_ratio = []
        args=[]
        for it_num in range(repeat_num):
            args.append([it_num, nodes_list, neighbor_dict, n_inf, n_take, beta, alpha, w, c0, cI, ita, gamma, mu, T])
    
        with Pool(10) as pool:
            results = pool.map(Gillespie_network, args)
        
        for res in results:
            infected_ratio.append(res[0])
            takemeasure_ratio.append(res[1])
        
        rho_I_dict[beta] = infected_ratio
        rho_take_dict[beta] = takemeasure_ratio
        
        #print(rho_I_dict)
        #print(rho_take_dict)
    np.save(_Data_PATH_XX_ + filename1 + '.npy', rho_I_dict)
    np.save(_Data_PATH_XX_ + filename2 + '.npy', rho_take_dict)



# In[ ]:


#结果存储
#np.save(_Data_PATH_XX_ + filename1 + '.npy', rho_I_dict)
#np.save(_Data_PATH_XX_ + filename2 + '.npy', rho_take_dict)

