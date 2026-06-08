import pandas as pd
import numpy as np

# ==================== 1. 读取你的数据 ====================
# 假设你的数据是Excel或CSV，每一行是一个区域（例如县/乡），列是各个指标
# 请把你的文件路径替换下面这行
df = pd.read_excel("D:\桌面应用\各地市POI基尼系数与核密度分析_2024.xlsx")   # 或 pd.read_csv('你的数据文件.csv')

# 如果你的列名和下面不一样，请先改成下面统一的名称（建议复制粘贴使用）
# 必须包含以下6个POI列（其他指标可以继续保留，不会影响）
required_columns = [
    '教育POI密度', '医疗POI密度', '养老POI密度', '文体POI密度',
    '交通POI密度', '商业POI密度'
]

# 检查列是否存在
missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"缺少以下列，请检查列名：{missing}")

# ==================== 2. Min-Max标准化（所有POI都是正向指标） ====================
def min_max_scale(series):
    return (series - series.min()) / (series.max() - series.min())

# 对6个POI分别标准化
poi_cols = required_columns  # 6类POI
for col in poi_cols:
    df[f'{col}_std'] = min_max_scale(df[col])

# ==================== 3. 按维度计算子指数（推荐做法） ====================
# 公共服务POI指数（4个指标）
public_poi_cols_std = [
    '教育POI密度_std', '医疗POI密度_std',
    '养老POI密度_std', '文体POI密度_std'
]
df['公共服务POI指数'] = df[public_poi_cols_std].mean(axis=1)   # 等权平均

# 基础设施POI指数（2个POI + 路网密度）
# 注意：路网密度不是POI，但你已把它放在基础设施里，这里一起算进去
# 如果你不想把路网密度放进来，把下面一行改成只取交通和商业两个_std即可
infra_poi_cols_std = [
    '交通POI密度_std', '商业POI密度_std',
    '路网密度_std'          # ← 如果你还没有标准化路网密度，先在上面加一行标准化
]
# 如果你还没有对路网密度做标准化，先加下面这行：
if '路网密度' in df.columns:
    df['路网密度_std'] = min_max_scale(df['路网密度'])

df['基础设施POI指数'] = df[infra_poi_cols_std].mean(axis=1)   # 等权平均

# ==================== 4. （可选）计算总POI指数 ====================
# 两个维度按重要性加权（可自行修改权重）
df['总POI指数'] = (
    0.55 * df['公共服务POI指数'] + 
    0.45 * df['基础设施POI指数']
)

# ==================== 5. 输出结果 ====================
# 保存带所有新指标的文件
df.to_excel('POI指数计算结果.xlsx', index=False)
print("✅ 计算完成！已保存到 'POI指数计算结果.xlsx'")
print(df[['公共服务POI指数', '基础设施POI指数', '总POI指数']].head())

# ==================== 如果你想改成加权平均（更推荐） ====================
# 把上面第3步和第4步替换成下面这段（示例权重，你可以自行调整）：
"""
# 公共服务POI指数（加权）
weights_public = {'教育POI密度_std': 0.25, '医疗POI密度_std': 0.30,
                  '养老POI密度_std': 0.20, '文体POI密度_std': 0.25}
df['公共服务POI指数'] = sum(df[col] * w for col, w in weights_public.items())

# 基础设施POI指数（加权）
weights_infra = {'交通POI密度_std': 0.40, '商业POI密度_std': 0.30, '路网密度_std': 0.30}
df['基础设施POI指数'] = sum(df[col] * w for col, w in weights_infra.items())

# 总POI指数
df['总POI指数'] = 0.55 * df['公共服务POI指数'] + 0.45 * df['基础设施POI指数']
"""














import pandas as pd
import numpy as np

# ==================== 1. 读取你的数据 ====================
# 请把下面路径改成你实际的Excel/CSV文件路径
df = pd.read_excel("D:\桌面应用\各地市POI基尼系数与核密度分析_2024.xlsx")   # 或 pd.read_csv('你的数据文件.csv')

# 你提供的12个POI列（基尼系数 + 核密度峰值）
gini_columns = [
    '教育POI基尼系数', '医疗POI基尼系数', '养老POI基尼系数',
    '商业POI基尼系数', '文体POI基尼系数', '交通POI基尼系数'
]

peak_columns = [
    '教育POI核密度峰值', '医疗POI核密度峰值', '养老POI核密度峰值',
    '商业POI核密度峰值', '文体POI核密度峰值', '交通POI核密度峰值'
]

all_poi_cols = gini_columns + peak_columns

# 检查列是否存在
missing = [col for col in all_poi_cols if col not in df.columns]
if missing:
    raise ValueError(f"❌ 缺少以下列，请检查列名是否完全一致：{missing}")
print("✅ 所有12个POI列已找到")

# ==================== 2. 标准化（重要！） ====================
# 基尼系数是负向指标（数值越大越不均衡 → 越差），核密度峰值是正向指标
def min_max_positive(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-8)

def min_max_negative(series):   # 基尼系数取反后标准化（数值越大越好）
    return (series.max() - series) / (series.max() - series.min() + 1e-8)

# 对核密度峰值做正向标准化
for col in peak_columns:
    df[f'{col}_std'] = min_max_positive(df[col])

# 对基尼系数做负向（取反）标准化
for col in gini_columns:
    df[f'{col}_std'] = min_max_negative(df[col])

# ==================== 3. 按维度计算子指数（完全按照你之前的分类） ====================
# 公共服务维度（教育、医疗、养老、文体 → 8个标准化指标）
public_std_cols = [
    '教育POI基尼系数_std', '教育POI核密度峰值_std',
    '医疗POI基尼系数_std', '医疗POI核密度峰值_std',
    '养老POI基尼系数_std', '养老POI核密度峰值_std',
    '文体POI基尼系数_std', '文体POI核密度峰值_std'
]

# 基础设施维度（商业 + 交通 → 4个标准化指标）
infra_std_cols = [
    '商业POI基尼系数_std', '商业POI核密度峰值_std',
    '交通POI基尼系数_std', '交通POI核密度峰值_std'
]

df['公共服务POI指数'] = df[public_std_cols].mean(axis=1)      # 等权平均
df['基础设施POI指数'] = df[infra_std_cols].mean(axis=1)        # 等权平均

# ==================== 4. 计算总POI指数 ====================
df['总POI指数'] = 0.55 * df['公共服务POI指数'] + 0.45 * df['基础设施POI指数']

# ==================== 5. 输出并保存 ====================
df.to_excel('POI指数计算结果.xlsx', index=False)

print("\n✅ 计算完成！已保存到文件：POI指数计算结果.xlsx")
print("新增的3个核心指标如下（前5行预览）：")
print(df[['公共服务POI指数', '基础设施POI指数', '总POI指数']].head())

print("\n各指标统计描述：")
print(df[['公共服务POI指数', '基础设施POI指数', '总POI指数']].describe())

# ==================== 如果你想改成加权平均（更推荐） ====================
# 把第3步和第4步替换成下面代码（示例权重，你可以自行修改）：
"""
# 公共服务POI指数（加权示例）
public_weights = {
    '教育POI基尼系数_std': 0.12, '教育POI核密度峰值_std': 0.13,
    '医疗POI基尼系数_std': 0.15, '医疗POI核密度峰值_std': 0.15,
    '养老POI基尼系数_std': 0.12, '养老POI核密度峰值_std': 0.12,
    '文体POI基尼系数_std': 0.10, '文体POI核密度峰值_std': 0.11
}
df['公共服务POI指数'] = sum(df[col] * w for col, w in public_weights.items())

# 基础设施POI指数（加权示例）
infra_weights = {
    '商业POI基尼系数_std': 0.25, '商业POI核密度峰值_std': 0.25,
    '交通POI基尼系数_std': 0.25, '交通POI核密度峰值_std': 0.25
}
df['基础设施POI指数'] = sum(df[col] * w for col, w in infra_weights.items())

df['总POI指数'] = 0.55 * df['公共服务POI指数'] + 0.45 * df['基础设施POI指数']
"""











































