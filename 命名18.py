import pandas as pd
import numpy as np
import os

# ====================== 1. 环境与数据配置 ======================
# 建议将文件路径替换为你的实际路径
input_file = r"D:\桌面应用\总统计表1(1)(3).xlsx"
output_result = "三省135县宜居性评价得分结果_最终版.xlsx"
output_weight = "指标权重分布表.xlsx"

# 定义指标体系：1代表正向指标，-1代表负向指标
# 这里严格对应你提供的最新列名
indicator_config = {
    # 生态环境维度
    "RSEI_mean": 1, "优良面积占比": 1, "NVDI": 1, "slope_mean": -1,
    # 公共服务维度
    "教育POI密度": 1, "医疗POI密度": 1, "每千人医院卫生院床位数": 1, "每千人社会福利收养性床位数": 1,
    # 经济发展维度
    "人均GDP": 1, "农村居民人均可支配收入": 1, "一般公共预算收入": 1,
    # 基础设施维度
    "路网密度": 1, "卫生厕所普及率": 1,
    # 空间可达性维度
    "教育POI基尼系数": -1, "医疗POI基尼系数": -1, "POI综合覆盖度": 1
}

features = list(indicator_config.keys())

# ====================== 2. 数据读取与预处理 ======================
try:
    df_raw = pd.read_excel(input_file)
    print(f"✅ 原始数据读取成功，样本量: {len(df_raw)}")
except Exception as e:
    print(f"❌ 读取失败，请检查文件路径或格式: {e}")
    exit()

# 强制转换数值型：解决 TypeError (float + str) 的核心步骤
for col in features:
    # errors='coerce' 会将无法转换的文字/空格变为 NaN
    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

# 缺失值处理：使用列均值填充，确保计算连续性
df = df_raw.copy()
df[features] = df[features].fillna(df[features].mean())

# ====================== 3. Min-Max 标准化 ======================
# 区分正负向指标，将数据统一映射至 [0, 1]
df_scaled = df.copy()

for col, direction in indicator_config.items():
    x_min = df[col].min()
    x_max = df[col].max()
    
    if x_max == x_min:
        df_scaled[col] = 0.0
    else:
        if direction == 1: # 正向指标
            df_scaled[col] = (df[col] - x_min) / (x_max - x_min)
        else: # 负向指标
            df_scaled[col] = (x_max - df[col]) / (x_max - x_min)

# ====================== 4. 熵值法 (Entropy Weight Method) ======================
def get_weights(data_norm):
    # 4.1 计算比重 P_ij，防止分母为0
    # 为避免 log(0)，在比重中加入极小值平移
    p = data_norm / (data_norm.sum(axis=0) + 1e-12)
    
    # 4.2 计算信息熵 E_j
    # 公式：E_j = -k * Σ(p_ij * ln(p_ij))，其中 k = 1/ln(n)
    n = len(data_norm)
    k = 1.0 / np.log(n)
    
    # 使用 numpy 掩码处理，只对大于0的值求ln
    p_log_p = p * np.log(p + 1e-12) 
    entropy = -k * p_log_p.sum(axis=0)
    
    # 4.3 计算冗余度 D_j 与 权重 W_j
    d = 1 - entropy
    w = d / d.sum()
    return w

weights = get_weights(df_scaled[features])

# 保存权重表以便检查
weight_df = pd.DataFrame({"指标": features, "方向": ["正向" if indicator_config[x]==1 else "负向" for x in features], "权重": weights.values})
weight_df.to_excel(output_weight, index=False)

# ====================== 5. 综合得分计算与排名 ======================
# 5.1 加权求和得到综合得分
df["综合得分"] = df_scaled[features].dot(weights)

# 5.2 排名计算
# 全局总排名
df["总排名"] = df["综合得分"].rank(ascending=False, method='min').astype(int)
# 省内排名（针对每个省单独计算）
df["省内排名"] = df.groupby("省名")["综合得分"].rank(ascending=False, method='min').astype(int)

# 5.3 结果整理：按得分从高到低排序
df_final = df.sort_values(by="综合得分", ascending=False)

# ====================== 6. 数据保存 ======================
try:
    # 导出包含所有原始信息、得分和排名的总表
    df_final.to_excel(output_result, index=False)
    print("\n" + "="*40)
    print("🚀 评价任务执行完毕！")
    print(f"1. 权重分析表已生成: {output_weight}")
    print(f"2. 综合排名表已生成: {output_result}")
    print("="*40)
except Exception as e:
    print(f"❌ 结果保存失败（请检查Excel是否被占用）: {e}")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
import pandas as pd
import numpy as np

# ====================== 1. 数据配置 ======================
input_file = r"D:\桌面应用\总统计表2024.xlsx"
output_result = "三省135县宜居性评价得分结果_最终版2.xlsx"

indicator_config = {
    "RSEI_mean": 1, "优良面积占比": 1, "NVDI": 1, "slope_mean": -1,
    "教育POI密度": 1, "医疗POI密度": 1, "每千人医院卫生院床位数": 1, "每千人社会福利收养性床位数": 1,
    "人均GDP": 1, "农村居民人均可支配收入": 1, "一般公共预算收入": 1,
    "路网密度": 1, "卫生厕所普及率": 1,
    "教育POI基尼系数": -1, "医疗POI基尼系数": -1, "POI综合覆盖度": 1
}
features = list(indicator_config.keys())

# ====================== 2. 读取与深度清洗 ======================
try:
    df_raw = pd.read_excel(input_file)
    # 去除表头或数据末尾可能存在的完全空白行
    df_raw = df_raw.dropna(how='all')
    print(f"✅ 原始数据读取成功，样本量: {len(df_raw)}")
except Exception as e:
    print(f"❌ 读取失败: {e}")
    exit()

# 【重要修复】确保“省名”和“县名”没有空值，否则无法进行分组排名
df_raw = df_raw.dropna(subset=["省名", "县名"])
df_raw["省名"] = df_raw["省名"].astype(str).str.strip()

# 强制转换数值型
for col in features:
    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

# 填充指标缺失值
df = df_raw.copy()
df[features] = df[features].fillna(df[features].mean())

# ====================== 3. 标准化与权重计算 ======================
df_scaled = df.copy()
for col, direction in indicator_config.items():
    x_min, x_max = df[col].min(), df[col].max()
    if x_max == x_min:
        df_scaled[col] = 0.0
    else:
        if direction == 1:
            df_scaled[col] = (df[col] - x_min) / (x_max - x_min)
        else:
            df_scaled[col] = (x_max - df[col]) / (x_max - x_min)

def get_weights(data_norm):
    p = data_norm / (data_norm.sum(axis=0) + 1e-12)
    n = len(data_norm)
    k = 1.0 / np.log(n)
    entropy = -k * (p * np.log(p + 1e-12)).sum(axis=0)
    d = 1 - entropy
    return d / d.sum()

weights = get_weights(df_scaled[features])

# ====================== 4. 得分与排名 (防报错处理) ======================
# 计算得分
df["综合得分"] = df_scaled[features].dot(weights)

# 【核心修复点】
# 1. 计算排名
# 2. 不使用 .astype(int)，而是使用 .astype('Int64') 或者干脆保持浮点数排名
# 这样即便有意外的空值，代码也不会崩溃
df["总排名"] = df["综合得分"].rank(ascending=False, method='min').astype('Int64')

# 分组排名：按省名进行省内排名
try:
    df["省内排名"] = df.groupby("省名")["综合得分"].rank(ascending=False, method='min').astype('Int64')
except Exception as e:
    print(f"⚠️ 省内排名计算警告: {e}")
    df["省内排名"] = np.nan

# ====================== 5. 结果保存 ======================
df_final = df.sort_values(by=["省名", "综合得分"], ascending=[True, False])

try:
    df_final.to_excel(output_result, index=False)
    print("\n" + "="*40)
    print("🚀 任务执行完毕！")
    print(f"结果文件: {output_result}")
    print("提示：如果排名列出现 <NA>，说明该行数据不完整。")
    print("="*40)
except Exception as e:
    print(f"❌ 保存失败: {e}")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    