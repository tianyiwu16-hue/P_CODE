# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 08:53:18 2024

@author: 20997
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# 创建3D图形对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 设置爱心的参数
t = np.linspace(0, 2 * np.pi, 100)
theta = np.linspace(0, 2 * np.pi, 100)
r = 1 - np.sin(theta)  # 这里的r可以控制爱心的形状

# 初始化爱心形状
x, y, z = [], [], []
for i in t:
    for j in theta:
        x.append(r * np.sin(j) * np.cos(i))
        y.append(r * np.sin(j) * np.sin(i))
        z.append(r * np.cos(j) + np.sin(i) * 0.5)  # 这里的0.5可以控制爱心的“厚度”

# 创建图形点对象
graph, = ax.plot(x, y, z, 'r-')

# 设置3D图形的范围
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-1.5, 1.5])

# 定义旋转函数
def rotate(frame):
    ax.view_init(elev=10., azim=frame)  # elev是仰角，azim是方位角

# 创建动画
ani = FuncAnimation(fig, rotate, frames=np.arange(0, 360, 1), interval=50)

# 展示动画
plt.show()