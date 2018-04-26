# -*- coding: utf-8 -*-
# Captain_N
import csv
import random
import numpy as np
import matplotlib .pyplot as plt

N = 769
start = 561
K=3#循环次数
M=10#每一循环的遍历次数
#读取数据
print("读取数据")
with open('links.csv', encoding='utf-8') as f:
    data = csv.reader(f)
    network =np.zeros((33312,2),dtype=np.int64)#int64是numpy中引入的一个类，即 numpy.int64
    for i, line in enumerate(data):#enumerate 将对象转为索引序列，可以同时获得索引和值。
        #print(line[0])  matlab中节点标号1~769，python中节点标号0~768.
        network[i][0]=int(line[0])-1
        network[i][1]=int(line[1])-1
    #print(network)
#存储为邻接矩阵
new_network =np.zeros((N,N),dtype=np.int64)
for i in range(33312):
    a= network[i][0]
    b= network[i][1]
    new_network[a][b]=1
#定义初始状态
print("定义初始状态")
state = np.zeros((N),dtype=np.int64)#初始化节点，1代表激活，0代表未激活
sentiment=np.zeros((N))
print(sentiment .shape)#(769,)
for n in range(N):
    sentiment[n]=random.random()#定义社交网络中节点对某一新闻的情感指数，随机独立
print(state.shape )#(769,)
state[start] = 1#使一个节点成为最初的传播节点
#起点备份，保证每次起点相同
old_state = state.copy()
old_sentiment = sentiment.copy()
#初始化循环10次传播30步遍历769节点的张量
c=np.zeros((K,M,N))
print(c.shape)
#开始循环
for i in range(K):
    c[i][0]=old_state.copy()
    #print(old_state)
    state= old_state.copy()
    new_state = state.copy()
    sentiment= old_sentiment.copy()
    activated=[start]
    activated_num=0
    print("开始循环")
    #开始传播
    for j in range(M-1):#29步后到达第30个状态
        new_activated = []
        print("开始第"+str(j)+"次传播")
        #遍历已激活节点
        for m in range(len(activated)):
            #print("开始遍历激活节点")
            #遍历所有节点
            for n in range(N):
                #print("开始遍历")
                # 判断该节点与已激活节点是否邻接
                if (new_network[activated[m]][n]==1):
                    #判断是否已经激活，如果激活则跳过
                    if(new_state[n]==0):
                        #判断是否被激活
                        if(random.random()<0.5*sentiment[n]):
                            new_state[n]=1
                            new_activated.append(n)

                    else:
                        continue

        c[i][j + 1] = new_state.copy()
        activated= new_activated.copy()
        activated_num+=len(new_activated)
    print("第"+str(i)+"次循环激活节点一共为："+str(activated_num+1)+"个")
#统计每次循环每步激活节点占比，准备画图
s=np.zeros((K,M))
#mean_s=np.zeros((K))
for i in range(K):
    for j in range (M):
        state =c[i][j]
        num =0
        for m in range(N):
            if (state[m]==1):
                num =num+1

        pdf = num/N
        s[i][j] = pdf
# mean_s[i]=s[i]
#出图
x_zhou=np.array(range (M))

plt.figure()
for i in range(K):
    plt.plot(x_zhou,s[i])
plt.ylim((0,1))
plt.xlim((0,10))
plt.xticks(np.linspace(0, 10, 11))#构建等差数列
plt.yticks(np.linspace(0, 1, 11))
plt.legend(labels = ['activated1','activated2','activated3'], loc = 'best')
plt.xlabel('Time/step')
plt.ylabel('Node density')



plt .show()


