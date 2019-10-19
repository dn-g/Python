import pandas as pd
import matplotlib.pyplot as plt
##因为matplotlib库里没有中文字体,所以这个可以解决坐标轴不能显示中文问题
from pylab import * 
mpl.rcParams['font.sans-serif'] = ['SimHei']  
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
#导入数据
data = pd.read_csv(r"C:\Users\86182\Desktop\K-means.txt",sep='\t',encoding='gbk')
#Z-score标准化
train_x = data[['2019国际排名','2018世界杯','2015亚洲杯']]
scaled_x= preprocessing.scale(train_x)
#k-means做聚类
from sklearn.cluster import KMeans
k= 3
"""
init: 初始簇中心的获取方法;n_init: 获取初始簇中心的更迭次数，为了弥补初始质心的影响，
算法默认会初始10次质心，实现算法，然后返回最好的结果。;max_iter: 最大迭代次数###
"""
model= KMeans(n_clusters= k, init='k-means++', n_init=10, max_iter= 300)
##返回各自文本的所被分配到的类索引
clf= model.fit(scaled_x)
pre_y= clf.predict(scaled_x)
#concat合并
result= pd.concat([data, pd.DataFrame(pre_y)], axis=1)
result
#####方法二画图
plt.savefig('my_fig.png')
plt.figure(figsize=(8,6))
ax = plt.subplot(projection = '3d')
ax.set_xlabel('2019-International-ranking',color = 'orange',fontsize = 16)
ax.set_ylabel('2018-world-cup',color = 'orange',fontsize = 16)
ax.set_zlabel('2015-asia-cup',color = 'orange',fontsize = 16)

# 绘制3d空间的点
ax.scatter3D(data['2019国际排名'],data['2018世界杯'],data['2015亚洲杯'],c =pre_y,s=90,alpha = 1)
