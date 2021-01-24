# -*- coding: utf-8 -*-
import numpy as np  
import matplotlib.mlab as mlab  
import matplotlib.pyplot as plt  
  
from scipy.stats import norm

# 数据  
mu = 100 # mean of distribution  
sigma = 15 # standard deviation of distribution  
x = mu + sigma * np.random.randn(10000)  
  
num_bins = 10
# the histogram of the data  
n, bins, patches = plt.hist(x, num_bins, histtype='stepfilled', density=0, stacked=0, facecolor='blue', alpha=1)  
 
# add a 'best fit' line  
# y=norm.pdf(bins,mu,sigma)

# plt.plot(bins, y, 'r--')  
plt.xlabel('Smarts')  
plt.ylabel('Probability')  
plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')  
  
# Tweak spacing to prevent clipping of ylabel  
plt.subplots_adjust(left=0.15)  
# plt.show()  
# plt.savefig("hist.jpg")

mean = 0
#标准差为1，反应数据集中还是分散的值
sigma = 1
x=mean+sigma*np.random.randn(10000)
fig,(ax0,ax1) = plt.subplots(nrows=2,figsize=(9,6))
#第二个参数是柱子宽一些还是窄一些，越大越窄越密
ax0.hist(x,40,density=1,histtype='bar',facecolor='yellowgreen',alpha=0.75)
##pdf概率分布图，一万个数落在某个区间内的数有多少个
ax0.set_title('pdf')
ax1.hist(x,20,density=1,histtype='bar',facecolor='pink',alpha=0.75,cumulative=True,rwidth=0.8)
#cdf累计概率函数，cumulative累计。比如需要统计小于5的数的概率
ax1.set_title("cdf")
fig.subplots_adjust(hspace=0.4)
plt.show()