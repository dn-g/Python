import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pylab import * 
mpl.rcParams['font.sans-serif'] = ['SimHei']  ##因为matplotlib库里没有中文字体,所以这个可以解决坐标轴不能显示中文问题
from sklearn import preprocessing
data = pd.read_table(r"C:\Users\86182\Desktop\K-means.txt",encoding='gbk')#导入数据
#Z-score标准化
train_x = data[['2006年世界杯','2007年亚洲杯','2010年世界杯']]
scaled_x= preprocessing.scale(train_x)
#k-means做聚类
from sklearn.cluster import KMeans
k= 3
model= KMeans(n_clusters= k, init='k-means++', n_init=10, max_iter= 300)
clf= model.fit(scaled_x)
pre_y= clf.predict(scaled_x)
#concat合并
result= pd.concat([data, pd.DataFrame(pre_y)], axis=1)
result
for i in range(0,10):
    if result[0][i]==0:
        plt.scatter(result["国别"][i],result[0][i],color='red')
    if result[0][i]==1:
        plt.scatter(result["国别"][i],result[0][i],color='black')
    if result[0][i]==2:
        plt.scatter(result["国别"][i],result[0][i],color='blue')
plt.show()
plt.savefig('my_fig.png')