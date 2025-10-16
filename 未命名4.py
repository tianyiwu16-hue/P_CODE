# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 08:55:00 2024

@author: 20997
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义一个函数来计算爱心上的点
def heart_shape(t, shrink=0.5):
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    x *= shrink
    y *= shrink
    z = np.linspace(0, 1, len(t))  # 简单的z坐标，用于形成曲面
    return x, y, z

# 创建图形和坐标轴
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 生成t值
t = np.linspace(0, 2 * np.pi, 1000)

# 计算爱心上的点
x, y, z = heart_shape(t)

# 绘制爱心曲面
ax.plot_surface(x, y, z, color='red', alpha=0.5)  # alpha控制透明度

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 由于matplotlib的3D动画限制，这里我们使用旋转视图来模拟动画效果
# 注意：以下代码实际上不会创建一个动画，而是展示如何旋转视图
# 如果你想要真正的动画效果，你可能需要考虑使用Mayavi或其他3D可视化库

# 旋转视图（仅展示如何旋转，不包含在动画中）
# ax.view_init(elev=10., azim=45)  # 你可以尝试改变这两个参数来旋转视图

# 展示图形
plt.show()

# 注意：如果你确实需要动画效果，你可能需要使用FuncAnimation来定期更新视图
# 但由于matplotlib的限制，这通常意味着旋转视图而不是更新图形的顶点数据

# 以下是使用FuncAnimation旋转视图的一个非常基础的示例框架
# 但请注意，这不会改变爱心的形状或大小，只会旋转视图
# def update(frame):
#     ax.view_init(elev=10., azim=frame)
#     ax.figure.canvas.draw_idle()
# 
# ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 1), interval=50)
# 但由于plot_surface不支持动态更新数据，上面的代码实际上不会按预期工作
# 对于动态3D图形，请考虑使用其他库。