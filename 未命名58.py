import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据
labels = ['专业度', '吸引力', '互动性', '品牌契合度', '传播力']  # 维度标签
num_vars = len(labels)

# 计算每个维度对应的角度（用于绘制多边形）
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# 确保多边形首尾相连（角度闭合）
angles += angles[:1]

# 2. 估计图片中的评分数据 (Mean 蓝色)
# 这些是根据原图蓝色区域相对于参考圆的位置估计的数值
# 假设参考圆 (Ref) 的值为 4.0
mean_values = [3.55, 4.31, 3.42, 4.18, 4.35]  # 对应 专业度 -> 传播力

# 确保多边形首尾相连（数据闭合）
mean_values += mean_values[:1]

# 3. 创建图表
fig, ax = plt.figure(figsize=(8, 8), dpi=100), plt.subplot(111, polar=True)

# 4. 绘制蓝色多边形区域 (Mean)
# 颜色使用 #B0C4DE (浅钢蓝)，不透明度 60%
ax.fill(angles, mean_values, color='#B0C4DE', alpha=0.6, label='Mean')

# 5. 绘制参考圆 (Ref)
# 假设参考值为 4.0
ref_value = 4.0
# 用 np.full 创建一个全为 ref_value 的数组，长度和 angles 相同
ref_values = np.full(num_vars + 1, ref_value)
# 颜色使用 #FFFACD (柠檬绸)，不透明度 50%
ax.fill(angles, ref_values, color='#FFFACD', alpha=0.5, label='Ref')

# 6. 设置坐标轴和刻度
# 设置 Y 轴范围（可根据数据调整）
ax.set_ylim(0, 5)

# 隐藏角度和径向刻度标签（使界面更简洁，像原图）
ax.set_xticklabels([])
ax.set_yticklabels([])

# 设置多边形顶点的径向网格线为虚线
ax.set_theta_zero_location('N')  # “专业度”在顶部 (N)
ax.set_theta_direction(-1)      # 顺时针方向排列

# 7. 添加维度标签
# angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False) # 重新计算用于设置刻度
for angle, label in zip(angles[:-1], labels):
    # 根据原图标签的位置微调
    if label == '专业度':
        y_label = 5.3
    elif label == '品牌契合度':
        y_label = 5.2
    else:
        y_label = 5.1

    # 绘制标签文本，中文需要设置字体（如果系统默认不支持）
    # 在这里假设已正确配置中文显示，否则可以使用 ax.set_xticklabels([u'专业度', ...])
    # 并指定字体，如 ax.tick_params(axis='x', labelsize=12, labelrotation=0)
    # 简单的方法是先不用中文测试，确认结构无误
    ax.text(angle, y_label, label,
            horizontalalignment='center',
            verticalalignment='center',
            fontfamily='SimHei',  # Windows下常用黑体，Mac/Linux可根据需要调整
            fontsize=12)

# 8. 添加图例
# 图例样式微调：使用小方块
legend = ax.legend(loc='upper right',
                   bbox_to_anchor=(1.15, 1.05),
                   handlelength=0.7,
                   frameon=False,  # 隐藏图例边框
                   fontsize=10)

# 修改图例项的句柄样式为方块
for item in legend.legend_handles:
    item.set_shape('s')

# 9. 设置标题
ax.set_title('花西子数字人评价各维度均值对比图',
             y=1.1,
             fontfamily='SimHei',
             fontsize=14)

# 显示图表
plt.tight_layout()








import matplotlib.pyplot as plt
import numpy as np

# 为了显示中文，需要设置字体（根据你的系统调整，Windows通常是SimHei）
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# 1. 准备数据
labels = ['专业度', '吸引力', '互动性', '品牌契合度', '传播力']  # 维度标签
num_vars = len(labels)

# 计算每个维度对应的角度（用于绘制多边形）
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# 确保多边形首尾相连（角度和数据闭合）
angles += angles[:1]

# 数据 (Mean 蓝色)
mean_values = [3.55, 4.31, 3.42, 4.18, 4.35]  # 对应 专业度 -> 传播力
mean_values += mean_values[:1]

# 2. 创建图表
fig, ax = plt.figure(figsize=(8, 8), dpi=100), plt.subplot(111, polar=True)

# 3. 绘制蓝色多边形区域 (Mean)
# 颜色使用 #B0C4DE (浅钢蓝)，不透明度 60%
ax.fill(angles, mean_values, color='#B0C4DE', alpha=0.6, label='Mean')

# 4. 绘制参考圆 (Ref)
ref_value = 4.0
ref_values = np.full(num_vars + 1, ref_value)
# 颜色使用 #FFFACD (柠檬绸)，不透明度 50%
ax.fill(angles, ref_values, color='#FFFACD', alpha=0.5, label='Ref')

# ==========================================
# ✨ 新增：手动添加数据标注 (Data Labels)
# ==========================================
# 我们只需遍历前 num_vars 个点（不需要闭合点）
for i in range(num_vars):
    # 获取当前点的角度和数值
    angle = angles[i]
    value = mean_values[i]
    
    # 将数值格式化为保留两位小数的字符串
    label_text = f"{value:.2f}"
    
    # 计算文字的放置位置（在数据点的基础上略微向外偏移）
    # 偏移量可以根据你的数据范围进行微调（例如 0.1 到 0.3 之间）
    text_r = value + 0.2 
    
    # 绘制文字标注
    # verticalalignment 和 horizontalalignment 用于确保文字不遮挡点
    ax.text(angle, text_r, label_text, 
            color='black',          # 文字颜色
            fontsize=11,            # 字体大小
            fontweight='bold',       # 加粗，使其更醒目
            horizontalalignment='center', # 水平居中
            verticalalignment='center')   # 垂直居中
# ==========================================

# 5. 设置坐标轴和网格
ax.set_ylim(0, 5) # 设置 Y 轴范围

# 隐藏默认的角度和径向刻度标签（使界面更简洁）
ax.set_xticklabels([])
ax.set_yticklabels([])

ax.set_theta_zero_location('N')  # “专业度”在顶部 (N)
ax.set_theta_direction(-1)      # 顺时针方向排列

# 6. 添加维度标签
for angle, label in zip(angles[:-1], labels):
    # 根据原图标签的位置微调
    if label == '专业度': y_label = 5.4
    elif label == '品牌契合度': y_label = 5.3
    else: y_label = 5.2

    ax.text(angle, y_label, label,
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=12)

# 7. 添加图例和标题
legend = ax.legend(loc='upper right',
                   bbox_to_anchor=(1.15, 1.05),
                   handlelength=0.7,
                   frameon=False, 
                   fontsize=10)

ax.set_title('花西子数字人评价各维度均值对比图',
             y=1.1,
             fontsize=14,
             fontweight='bold')

# 显示图表
plt.tight_layout()
plt.show()













plt.show()