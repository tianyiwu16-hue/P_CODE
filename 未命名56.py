import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体，避免中文显示乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 1. 准备数据
dimensions = ['专业度', '吸引力', '互动性', '品牌契合度', '传播力']
scores = [3.55, 4.31, 3.42, 4.18, 4.35]

# 2. 定义颜色方案（突出最高/最低维度）
# 初始颜色设为中性灰
colors = ['#6c757d'] * len(dimensions)
# 找到最高分（传播力、吸引力）和最低分（互动性、专业度）的索引
max_indices = [1, 4]  # 吸引力、传播力
min_indices = [0, 2]  # 专业度、互动性
# 最高分设为品牌红，最低分设为浅灰，其余为蓝色
for idx in max_indices:
    colors[idx] = '#e63946'  # 红色突出最高分
for idx in min_indices:
    colors[idx] = '#adb5bd'  # 浅灰突出最低分

# 3. 创建画布和子图（设置合适的尺寸，适配报告）
fig, ax = plt.subplots(figsize=(10, 6))

# 4. 绘制柱状图
bars = ax.bar(dimensions, scores, color=colors, width=0.6, edgecolor='#495057', linewidth=0.8)

# 5. 设置坐标轴和标题
ax.set_title('花西子数字人评价各维度均值对比图（Z世代消费者）', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('评价维度', fontsize=12, labelpad=10)
ax.set_ylabel('评分均值（1-5分）', fontsize=12, labelpad=10)

# 设置Y轴范围（符合李克特量表1-5分）
ax.set_ylim(0, 5)
# 添加Y轴刻度线，每0.5分一个刻度，更精细
ax.set_yticks(np.arange(0, 5.5, 0.5))

# 6. 在柱子顶部标注具体均值
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
            f'{height:.2f}',  # 保留两位小数，和原始数据一致
            ha='center', va='bottom', fontsize=11, fontweight='medium')

# 7. 美化图表（移除顶部和右侧边框，添加网格线）
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='#999')  # 添加Y轴网格线，增强可读性
ax.set_axisbelow(True)  # 网格线置于柱子下方

# 8. 调整布局，避免元素重叠
plt.tight_layout()

# 9. 保存图片（高清格式，适合插入报告）
plt.savefig('花西子数字人评价维度对比图.png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()












import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch  # 导入圆角矩形类

# ===================== 全局设置 =====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 兼容中英文
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Arial'  # 学术图表常用字体
plt.rcParams['axes.linewidth'] = 0.8   # 坐标轴线条宽度

# ===================== 基础参数 =====================
fig, ax = plt.subplots(figsize=(12, 9), dpi=150)  # 更大尺寸，更高分辨率
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')  # 隐藏坐标轴

# 定义专业配色（莫兰迪色系，更适合学术场景）
colors = {
    'input': '#8eb897',      # 浅绿（数字人代言特征）
    'cognition': '#e5b099',  # 浅橙（品牌认知）
    'affect': '#b88686',     # 浅红（品牌情感）
    'behavior': '#8ea8c3'    # 浅蓝（消费行为）
}

# 定义箭头样式
arrow_style = dict(arrowstyle='->', linewidth=1.8, color='#333333')
arrow_style_highlight = dict(arrowstyle='->', linewidth=2.2, color='#8b0000', linestyle='-')

# ===================== 绘制模块 =====================
# 1. 数字人代言特征（输入层）- 带圆角和阴影效果
input_rect = FancyBboxPatch((1, 7.5), 3, 2, 
                          facecolor=colors['input'], edgecolor='#2f4f4f', linewidth=1.2,
                          boxstyle="round,pad=0.1")  # 圆角矩形
ax.add_patch(input_rect)
# 模块标题和内容
ax.text(2.5, 9.2, '数字人代言特征', ha='center', va='center', fontsize=12, fontweight='bold', color='#2f4f4f')
features = ['专业度 (3.55)', '吸引力 (4.31)', '互动性 (3.42)', '品牌契合度 (4.18)', '传播力 (4.35)']
for i, feat in enumerate(features):
    ax.text(2.5, 8.9 - i*0.3, feat, ha='center', va='center', fontsize=10, color='#333333')

# 2. 品牌认知（认知层）- 带圆角
cognition_rect = FancyBboxPatch((4, 6), 4, 2.5, 
                              facecolor=colors['cognition'], edgecolor='#2f4f4f', linewidth=1.2,
                              boxstyle="round,pad=0.1")
ax.add_patch(cognition_rect)
ax.text(6, 8.2, '品牌认知 (Cognition)', ha='center', va='center', fontsize=12, fontweight='bold', color='#2f4f4f')
cognitions = ['品牌认知: 4.25', '品牌质量: 3.91', '品牌形象: 4.42']
for i, cog in enumerate(cognitions):
    ax.text(6, 7.9 - i*0.3, cog, ha='center', va='center', fontsize=10, color='#333333')

# 3. 品牌情感（情感层）- 带圆角
affect_rect = FancyBboxPatch((4, 3), 4, 2.5, 
                           facecolor=colors['affect'], edgecolor='#2f4f4f', linewidth=1.2,
                           boxstyle="round,pad=0.1")
ax.add_patch(affect_rect)
ax.text(6, 5.2, '品牌情感 (Affect)', ha='center', va='center', fontsize=12, fontweight='bold', color='#2f4f4f')
affects = ['品牌吸引力: 4.10', '品牌信任: 3.68']
for i, aff in enumerate(affects):
    ax.text(6, 4.9 - i*0.3, aff, ha='center', va='center', fontsize=10, color='#333333')

# 4. 消费行为（行为层）- 带圆角
behavior_rect = FancyBboxPatch((4, 0.5), 4, 2, 
                             facecolor=colors['behavior'], edgecolor='#2f4f4f', linewidth=1.2,
                             boxstyle="round,pad=0.1")
ax.add_patch(behavior_rect)
ax.text(6, 2.2, '消费行为 (Behavior)', ha='center', va='center', fontsize=12, fontweight='bold', color='#2f4f4f')
behaviors = ['购买意愿', '推荐意愿']
for i, beh in enumerate(behaviors):
    ax.text(6, 1.9 - i*0.3, beh, ha='center', va='center', fontsize=10, color='#333333')

# ===================== 绘制路径箭头 =====================
# 路径1：数字人代言 → 品牌认知（高亮）
ax.annotate('', xy=(4, 8.2), xytext=(3.8, 8.2), arrowprops=arrow_style_highlight)
ax.text(2.2, 8.8, '路径1\n数字人代言→品牌认知', ha='center', va='center', 
        fontsize=10, color='#8b0000', fontweight='medium',
        bbox=dict(boxstyle="round,pad=0.2", facecolor='#fff5f5', edgecolor='#8b0000'))

# 路径2：品牌认知 → 品牌情感
ax.annotate('', xy=(6, 5.5), xytext=(6, 6), arrowprops=arrow_style)
ax.text(7.2, 6.5, '路径2\n品牌认知→品牌情感', ha='center', va='center', 
        fontsize=10, color='#2f4f4f',
        bbox=dict(boxstyle="round,pad=0.2", facecolor='#f8f9fa', edgecolor='#2f4f4f'))

# 路径3：品牌情感 → 消费行为
ax.annotate('', xy=(6, 3), xytext=(6, 3.5), arrowprops=arrow_style)
ax.text(7.2, 4.2, '路径3\n品牌情感→消费行为', ha='center', va='center', 
        fontsize=10, color='#2f4f4f',
        bbox=dict(boxstyle="round,pad=0.2", facecolor='#f8f9fa', edgecolor='#2f4f4f'))

# 路径4：直接情感影响（弧形箭头，高亮）
ax.annotate('', xy=(4.2, 5.5), xytext=(3.8, 8), 
            arrowprops={**arrow_style_highlight, 'connectionstyle': 'arc3,rad=0.3'})
ax.text(1.8, 5.5, '路径4\n直接情感影响', ha='center', va='center', 
        fontsize=10, color='#8b0000', fontweight='medium',
        bbox=dict(boxstyle="round,pad=0.2", facecolor='#fff5f5', edgecolor='#8b0000'))

# 标注“数字人刺激”
ax.text(5.5, 8.5, '↑ 数字人刺激', ha='center', va='center', fontsize=10, 
        color='#2f4f4f', fontweight='medium')

# ===================== 标题和布局 =====================
# 主标题（更规范的学术格式）
plt.title('花西子数字人代言品牌印象形成机制\nABC态度模型核心路径（Z世代消费者 N=710）',
          fontsize=14, fontweight='bold', pad=30, color='#2f4f4f')

# 添加网格背景（浅灰色，提升层次感）
ax.add_patch(FancyBboxPatch((0, 0), 10, 10, facecolor='#fafafa', edgecolor='none', zorder=-1))

# 调整布局，避免内容溢出
plt.tight_layout()

# 保存高清图片（支持透明背景）
plt.savefig('花西子数字人代言品牌印象形成机制_优化版.png', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()























