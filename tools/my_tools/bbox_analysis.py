import pandas as pd
import seaborn as sns
import numpy as np
import json
import matplotlib.pyplot as plt
from math import sqrt

# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['font.family']='sans-serif'
# plt.rcParams['figure.figsize'] = (6.0, 6.0)


# 读取数据
# ann_json = 'instances_train2017.json'
ann_json = '/home/sun/projects/mmdetection/data/SSDD/SSDD_coco/annotations/instances_sarship_train.json'
ann_json = '/home/sun/projects/mmdetection/data/SSDD/SSDD_coco/annotations/instances_sarship_test.json'
with open(ann_json) as f:
    ann=json.load(f)

#################################################################################################
#创建类别标签字典
category_dic=dict([(i['id'],i['name']) for i in ann['categories']])
counts_label=dict([(i['name'],0) for i in ann['categories']])
for i in ann['annotations']:
    counts_label[category_dic[i['category_id']]]+=1

# 标注长宽高比例
box_w = []
box_h = []
box_wh = []
size = []
for a in ann['annotations']:
    # if a['category_id'] != 0:
    box_w.append(round(a['bbox'][2],2))
    box_h.append(round(a['bbox'][3],2))
    # 长边 / 短边
    wh = round(a['bbox'][2]/a['bbox'][3],0)
    if wh <1 :
        wh = round(a['bbox'][3]/a['bbox'][2],0)
    box_wh.append(wh)
    size.append(round(sqrt(a['area']), 0))



# 所有标签的长宽高比例
box_wh_unique = sorted(list(set(box_wh)))
box_wh_count=[box_wh.count(i) for i in box_wh_unique]
total = len(size)  # train: 2007, test: 544
box_wh_ratio = [i / total for i in box_wh_count]
print(box_wh_ratio[0] + box_wh_ratio[1])  # 0.8076731439960139

size_unique = sorted(list(set(size)))
size_count=[size.count(i) for i in size_unique]


num_bins = []
for i in range(9):
    # gap = (2 ** (i + 1) - 2 ** i)
    # num_bins += [2 ** i, 2 ** i + gap * 0.25, 2 ** i + gap * 0.5, 2 ** i + gap * 0.75]
    num_bins += [2 ** i, 2 ** i * sqrt(2)]
    # num_bins += [2 ** i]
num_bins.pop(-1)
xtick = [int(2 ** i) for i in range(9)]
xtick_label_a = [str(i) for i in xtick]
xtick_label_b = [str(int(i)) + ':' + '1' for i in box_wh_unique]

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].hist(size, num_bins, histtype='bar', density=0, stacked=1, facecolor='blue', alpha=1, edgecolor='black', lw=1)
axs[0].set_xscale('log', basex=sqrt(2))
axs[0].set_xticks(xtick)
axs[0].set_xticklabels(xtick_label_a)
axs[0].set_xlim(4, 256)
axs[0].set_ylim(0, 600)
axs[0].set_xlabel('(a) ground-truth box size (pixel)')
axs[0].set_ylabel('#count')

axs[1].bar(box_wh_unique, box_wh_count, width=0.5, color='b', edgecolor='black', lw=1)
axs[1].set_xlabel('(b) ground-truth box relative aspect ratio')
axs[1].set_ylabel('#count')
axs[1].set_xticks(box_wh_unique)
axs[1].set_xticklabels(xtick_label_b, rotation=0)
# axs[1].set_xlim(0, 8)
axs[1].set_ylim(0, 1000)

fig.tight_layout()
# plt.subplots_adjust(left=0.15)  
plt.show()


###
fig, ax = plt.subplots()
n, bins, patches = plt.hist(size, num_bins, histtype='bar', density=0, stacked=1, facecolor='blue', alpha=1)
cumsum = 0
for s, ratio in zip(bins, n):
    print('{:.2f}'.format(s), '{:.2f}'.format(ratio / total * 100))
    if 16 <= s < 128:
        cumsum += ratio / total * 100
print(cumsum)
print(bins)
print(n * 100)
ax.set_xscale('log', base=2)
plt.xticks(xtick, xtick)
plt.xlim(4, 256)
plt.ylim(0, 600)
# plt.xlabel('Smarts')  
# plt.ylabel('Probability')  
ax.set_xlabel('ground-truth box size (pixel)')
ax.set_ylabel('#count')
# ax.grid()
# plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')  
# Tweak spacing to prevent clipping of ylabel  
plt.subplots_adjust(left=0.15)  
plt.show()  

# 绘图
# wh_df = pd.DataFrame(box_wh_count,index=box_wh_unique,columns=['宽高比数量'])
# # wh_df.plot(kind='bar',color="#55aacc",legend=False)
# ax = wh_df.plot(kind='bar',color='b',legend=False)
# ax.set_xlabel('ground-truth box relative aspect ratio')
# ax.set_ylabel('#count')
# plt.show()  

fig, ax = plt.subplots()
ax.bar(box_wh_unique, box_wh_count, color='b')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('ground-truth box relative aspect ratio')
ax.set_ylabel('#count')
ax.set_xticks(box_wh_unique)
# ax.set_xticklabels(labels)
ax.set_ylim(0, 1000)
fig.tight_layout()
plt.show()  





