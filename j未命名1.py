# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 08:28:25 2024

@author: 20997
"""

import turtle
import math

# 初始化屏幕
screen = turtle.Screen()
screen.title('动的爱心')
screen.bgcolor('black')

# 初始化海龟
heart = turtle.Turtle()
heart.color('red')
heart.fillcolor('red')
heart.speed(0)  # 设置最快速度
heart.penup()

# 画爱心的函数
def draw_heart(size, x, y):
    heart.goto(x, y)  # 移动到指定位置
    heart.pendown()
    heart.begin_fill()
    heart.left(50)
    heart.forward(133 * size)
    heart.circle(50 * size, 200)
    heart.right(140)
    heart.circle(50 * size, 200)
    heart.forward(133 * size)
    heart.end_fill()
    heart.penup()

# 动态改变爱心大小并移动
def animate_heart():
    size = 0.1
    angle = 0
    frame_count = 0  # 添加帧计数器
    max_frames = 300  # 设置最大帧数
    while frame_count < max_frames:
        heart.clear()
        x = math.sin(angle) * 100  # 计算x坐标
        y = math.cos(angle) * 100 - 150  # 计算y坐标
        draw_heart(size, x, y)
        size += 0.05  # 增大爱心大小
        if size > 1:  # 当大小超过1时重置
            size = 0.1
        angle += 0.05  # 改变角度以移动爱心位置
        frame_count += 1  # 更新帧计数器
        
        # 延迟以控制动画速度
        screen.update()  # 更新屏幕显示

    # 动画结束后关闭窗口
    screen.bye()

# 开始动画
animate_heart()

# 在这个例子中，不需要调用turtle.done()，因为animate_heart中的循环会控制窗口的关闭
