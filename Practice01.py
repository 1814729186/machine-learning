# -*- coding: utf-8 -*-
#program by Ma Zhongping
#2020.05.03
#调用sklearn model实现可视化KNN，参考代码网站https://scikit-learn.org
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
##from PIL import Image

point = []
label = []
cur_label = 0
cur_marker = '.'
cur_color = 'orange'

n_neighbors = 0


def _drawmap(): 
    global cur_label
    global cur_marker
    global cur_color
    global point
    global label
    global num
    global _weights
    cmap_light = ListedColormap(['orange','cyan'])
    cmap_bold = ListedColormap(['darkorange', 'c'])
    plt.close(1)
    print("Calculating......")
    for n_neighbors in num:
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=_weights)
        point = np.array(point)
        #创建分类器并训练数据
        clf.fit(point, label)
        h = 0.1  # 填色分辨率
        x_min=0.0
        x_max = 100.0
        y_min, y_max = 0.0,100.0
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
        plt.scatter(point[:, 0], point[:, 1], c=label, cmap=cmap_bold,
                    edgecolor='k', s=20)
        plt.title("2-Class classification (k = %i ,weigth = %s)" % (n_neighbors,_weights))
        print("k = %i ok"%(n_neighbors))
    plt.show()

def on_press(event):
    #右键表示输入结束
    global cur_label
    global cur_marker
    global cur_color
    
    if cur_label <= 1:
        if str(event.button) == 'MouseButton.RIGHT':
            cur_label +=1
        if cur_label >= 2:
            print(point)
            print(label)
        if cur_label != 0:
            cur_marker = '.'
            cur_color = 'c'
            plt.title('Class 2')
        if cur_label > 1:
            _drawmap()

        print("my position:" ,event.button,event.xdata, event.ydata)
        #将创建的点在途中表示出来
        if str(event.button) == 'MouseButton.LEFT':
            plt.scatter(event.xdata,event.ydata,color = cur_color,edgecolor='k',s = 20)
            #将点放入列表中
            point.append([int(event.xdata),int(event.ydata)])
            label.append(cur_label)
        fig.show()

num = list(map(int, input("输入k:(可输入多个，空格隔开)\n").strip().split()))
_weights = 'distance'    #距离计算方式
ch = input("输入选用的距离计算方式：（1-欧氏距离，2-曼哈顿距离）\n")
if ch == '2':
    _weights = 'uniform'
ch = input("提示：\t打开图表窗口后点击鼠标左键添加第一类点.\n\t点击一次右键再点击多次左键添加第二类点.\n\t再次点击右键完成所有点的创建")
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.set(xlim = [0,100],ylim = [0,100],title ='Class 1',ylabel = 'Y_Axis',xlabel = 'X_Axis')
fig.canvas.mpl_connect('button_press_event', on_press)
plt.show()