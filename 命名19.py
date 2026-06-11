import pandas as pd
import numpy as np

# ====================== 1. 数据配置 ======================
input_file = r"D:\桌面应用\总统计表2024.xlsx"
output_result = "三省135县宜居性评价得分结果_最终版3.xlsx"

indicator_config = {
    "RSEI_mean": 1, "优良面积占比": 1, "NVDI": 1, "slope_mean": -1,
    "教育POI密度": 1, "医疗POI密度": 1,"文体POI密度": 1, "每千人医院卫生院床位数": 1, "每千人社会福利收养性床位数": 1,
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
    
    
    
    
    
    
    
    
    
    

    

import pandas as pd
import numpy as np

# ====================== 1. 数据配置与读取 ======================
input_file = r"D:\桌面应用\总统计表2024.xlsx"
output_result = "三省135县宜居性评价得分及检验结果.xlsx"

# 定义指标体系（已加入文体和商业POI密度）
indicator_config = {
    # 生态环境
    "RSEI_mean": 1, "优良面积占比": 1, "NVDI": 1, "slope_mean": -1,
    # 公共服务
    "教育POI密度": 1, "医疗POI密度": 1, "文体POI密度": 1, "商业POI密度": 1,
    "每千人医院卫生院床位数": 1, "每千人社会福利收养性床位数": 1,
    # 经济发展
    "人均GDP": 1, "农村居民人均可支配收入": 1, "一般公共预算收入": 1,
    # 基础设施
    "路网密度": 1, "卫生厕所普及率": 1,
    # 空间可达性
    "教育POI基尼系数": -1, "医疗POI基尼系数": -1, "POI综合覆盖度": 1
}
features = list(indicator_config.keys())

try:
    df_raw = pd.read_excel(input_file).dropna(how='all')
    df_raw = df_raw.dropna(subset=["省名", "县名"]) # 剔除关键信息缺失行
    print(f"✅ 数据读取成功，样本量: {len(df_raw)}")
except Exception as e:
    print(f"❌ 读取失败: {e}")
    exit()

# ====================== 2. 数据预处理 ======================
# 1. 强制数值化
for col in features:
    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

# 2. 填充缺失值（均值填充）
df = df_raw.copy()
df[features] = df[features].fillna(df[features].mean())

# ====================== 3. 专项相关性检验 ======================
# 针对 文体POI密度 和 商业POI密度 做相关性检验
col_a = "文体POI密度"
col_b = "商业POI密度"

# 计算 Pearson 相关系数
correlation_value = df[col_a].corr(df[col_b])

print("\n" + "="*30)
print(f"📊 专项指标相关性检验结果:")
print(f"指标 A: {col_a}")
print(f"指标 B: {col_b}")
print(f"相关系数 (Pearson r): {correlation_value:.4f}")

# 检验评价
if abs(correlation_value) > 0.9:
    print("📢 评价：相关性极高（>0.9），建议考虑指标是否存在统计学冗余。")
elif abs(correlation_value) > 0.7:
    print("📢 评价：高度相关，说明商业与文体设施分布具有高度一致性。")
else:
    print("📢 评价：相关性适中，两个指标具有较好的独立代表性。")
print("="*30 + "\n")

# ====================== 4. 熵值法综合评价 ======================
# 4.1 标准化 (Min-Max)
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

# 4.2 计算权重
def get_ewm_weights(data_norm):
    p = data_norm / (data_norm.sum(axis=0) + 1e-12)
    n = len(data_norm)
    k = 1.0 / np.log(n)
    entropy = -k * (p * np.log(p + 1e-12)).sum(axis=0)
    d = 1 - entropy
    return d / d.sum()

weights = get_ewm_weights(df_scaled[features])

# 4.3 计算综合得分与排名
df["宜居性得分"] = df_scaled[features].dot(weights)
# 使用 Int64 允许整数排名包含空值，防止报错
df["总排名"] = df["宜居性得分"].rank(ascending=False, method='min').astype('Int64')
df["省内排名"] = df.groupby("省名")["宜居性得分"].rank(ascending=False, method='min').astype('Int64')

# ====================== 5. 结果保存 ======================
weight_df = pd.DataFrame({
    "指标": features,
    "方向": ["正向" if indicator_config[x]==1 else "负向" for x in features],
    "最终权重": weights.values
})

# 创建专项检验结果表
test_res_df = pd.DataFrame({
    "检验对象": [f"{col_a} vs {col_b}"],
    "相关系数": [correlation_value],
    "结论": ["高度冗余" if abs(correlation_value) > 0.9 else "具有独立性"]
})

# 按排名排序
df_final = df.sort_values(by="总排名")

with pd.ExcelWriter(output_result) as writer:
    df_final.to_excel(writer, sheet_name="宜居性得分排名", index=False)
    weight_df.to_excel(writer, sheet_name="指标权重分配", index=False)
    test_res_df.to_excel(writer, sheet_name="商业文体相关性检验", index=False)

print(f"🚀 计算完成！最终结果已保存至: {output_result}")
    
    
    
    
    
    
    
    
    
    
    
    
    
# 提取这两个指标并计算相关系数矩阵
medical_corr = df[['医疗POI密度', "每千人医院卫生院床位数"]].corr()

# 获取具体的 Pearson 相关系数值
r_value = medical_corr.loc['医疗POI密度', "每千人医院卫生院床位数"]

print("="*40)
print("🏥 医疗资源相关性专项检验")
print(f"指标 1: 医疗POI密度 (设施覆盖面)")
print(f"指标 2: 每千人医疗床位数 (资源承载力)")
print(f"Pearson 相关系数 (r): {r_value:.4f}")
print("-" * 40)

# 自动评价
if abs(r_value) > 0.8:
    print("📢 评价：极强相关。说明医疗点位的增多与床位规模高度同步。")
elif abs(r_value) > 0.5:
    print("📢 评价：中等程度相关。两者具有一致性，但各具侧重点。")
else:
    print("📢 评价：弱相关。说明医疗点位多并不代表床位资源充足，两个指标互补性强。")
print("="*40)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# ====================== 权重检验逻辑 ======================

# 1. 基础排序显示
weights_sorted = pd.DataFrame({
    "指标": features,
    "权重": weights.values
}).sort_values(by="权重", ascending=False)

print("="*40)
print("⚖️ 权重分布排序 (前5名与后5名)")
print(weights_sorted.head(5))
print("...")
print(weights_sorted.tail(5))

# 2. 维度权重汇总 (将 18 个指标归类为 5 个维度进行检验)
dimension_map = {
    "生态环境": ["RSEI_mean", "优良面积占比", "NVDI", "slope_mean"],
    "公共服务": ["教育POI密度", "医疗POI密度", "文体POI密度",  "每千人医院卫生院床位数", "每千人社会福利收养性床位数"],
    "经济发展": ["人均GDP", "农村居民人均可支配收入", "一般公共预算收入"],
    "基础设施": ["路网密度", "卫生厕所普及率","商业POI密度",],
    "空间可达性": ["教育POI基尼系数", "医疗POI基尼系数", "POI综合覆盖度"]
}

dim_weights = {}
for dim, cols in dimension_map.items():
    # 提取当前维度在 weights 序列中的值并求和
    dim_weights[dim] = weights[cols].sum()

dim_weight_df = pd.DataFrame(list(dim_weights.items()), columns=['维度', '总权重']).sort_values(by='总权重', ascending=False)

print("\n" + "="*40)
print("🌍 评价维度贡献度分析")
print(dim_weight_df)

# 3. 权重合理性检验 (偏离度检查)
max_w = weights.max()
min_w = weights.min()
mean_w = weights.mean()
std_w = weights.std()

print("\n" + "="*40)
print("🔍 权重数理逻辑检验报告")
print(f"1. 权重加总: {weights.sum():.2f} (应等于 1.0)")
print(f"2. 最大权重偏离: {max_w/mean_w:.2f} 倍均值 (建议不超过 5 倍)")
print(f"3. 变异系数 (Std/Mean): {std_w/mean_w:.4f} (反映权重分配的均衡性)")

if max_w > 0.3:
    print("⚠️ 风险提示：存在权重过大的指标（>30%），建议检查该指标是否存在极端异常值！")
else:
    print("✅ 结果评估：权重分布相对均衡，未出现单一指标统治排名的情况。")
print("="*40)
    










import pandas as pd
import numpy as np

# ====================== 1. 配置与维度划分 ======================
input_file = r"D:\桌面应用\总统计表2024.xlsx"
output_file = "三省135县宜居性评价_最终版.xlsx"

# 更新后的维度划分
dimension_map = {
    "生态环境": ["RSEI_mean", "优良面积占比", "NVDI", "slope_mean"],
    "公共服务": ["教育POI密度", "医疗POI密度", "文体POI密度", "每千人医院卫生院床位数", "每千人社会福利收养性床位数"],
    "经济发展": ["人均GDP", "农村居民人均可支配收入", "一般公共预算收入"],
    "基础设施": ["路网密度", "卫生厕所普及率", "商业POI密度"],
    "空间可达性": ["教育POI基尼系数", "医疗POI基尼系数", "POI综合覆盖度"]
}

# 自动获取指标列表
all_features = [col for cols in dimension_map.values() for col in cols]

# 指标方向定义 (1:正向, -1:负向)
direction_map = {
    "RSEI_mean": 1, "优良面积占比": 1, "NVDI": 1, "slope_mean": -1,
    "教育POI密度": 1, "医疗POI密度": 1, "文体POI密度": 1, "每千人医院卫生院床位数": 1, "每千人社会福利收养性床位数": 1,
    "人均GDP": 1, "农村居民人均可支配收入": 1, "一般公共预算收入": 1,
    "路网密度": 1, "卫生厕所普及率": 1, "商业POI密度": 1,
    "教育POI基尼系数": -1, "医疗POI基尼系数": -1, "POI综合覆盖度": 1
}

# ====================== 2. 数据读取与预处理 ======================
df_raw = pd.read_excel(input_file).dropna(how='all')
df_raw = df_raw.dropna(subset=["省名", "县名"])
for col in all_features:
    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
df = df_raw.copy()
df[all_features] = df[all_features].fillna(df[all_features].mean())

# ====================== 3. 计算逻辑 ======================
def calculate_ewm(data):
    p = data / (data.sum(axis=0) + 1e-12)
    n = len(data)
    k = 1.0 / np.log(n)
    entropy = -k * (p * np.log(p + 1e-12)).sum(axis=0)
    d = 1 - entropy
    return d / (d.sum() + 1e-12)

# 标准化与分层计算
df_scaled = df.copy()
final_weights_series = pd.Series(dtype=float)

for dim, cols in dimension_map.items():
    # 标准化
    for col in cols:
        x_min, x_max = df[col].min(), df[col].max()
        if x_max == x_min: df_scaled[col] = 0.0
        else:
            direction = direction_map[col]
            df_scaled[col] = (df[col] - x_min) / (x_max - x_min) if direction == 1 else (x_max - df[col]) / (x_max - x_min)
    
    # 维度内权重计算并平滑 (各维度分配20%权重)
    internal_w = calculate_ewm(df_scaled[cols])
    final_weights_series = pd.concat([final_weights_series, internal_w * 0.2])

# 最终权重平滑 (80% 熵权 + 20% 均权)
w_equal = 1.0 / len(all_features)
final_weights_smoothed = 0.8 * final_weights_series + 0.2 * w_equal

# ====================== 4. 得分、排名与权重输出 ======================
df["综合得分"] = df_scaled[all_features].dot(final_weights_smoothed)
df["总排名"] = df["综合得分"].rank(ascending=False, method='min').astype('Int64')
df["省内排名"] = df.groupby("省名")["综合得分"].rank(ascending=False, method='min').astype('Int64')

# 生成权重对照表
weight_table = pd.DataFrame({
    "指标名称": all_features,
    "最终权重": final_weights_smoothed.values
}).sort_values(by="最终权重", ascending=False)

# ====================== 5. 保存结果 ======================
with pd.ExcelWriter(output_file) as writer:
    df.sort_values(by="综合得分", ascending=False).to_excel(writer, sheet_name="宜居性总得分表", index=False)
    weight_table.to_excel(writer, sheet_name="权重对照表", index=False)

print("✅ 计算完成！结果已保存。")
print(weight_table.head(10)) # 控制台预览前10名权重





import pandas as pd

# 读取刚才生成的 Excel 文件
df_new = pd.read_excel("D:\桌面应用\三省135县宜居性评价_总得分.xlsx")

# 查看前几行数据
print(df_new.head())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体（确保图表显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# ====================== 1. 统计分析计算 ======================
# 假设 df 是上一段代码计算完得分后的结果表
# stats_df = df.groupby("省名")["综合得分"].agg(['mean', 'std', 'max', 'min']).reset_index()
stats_df = df.groupby("省名")["综合得分"].agg(['mean', 'std', 'max', 'min']).rename(
    columns={'mean': '均值', 'std': '标准差', 'max': '最高分', 'min': '最低分'}
)

print("📊 三省宜居性评价统计分析结果：")
print(stats_df)
stats_df.to_excel("D:\桌面应用\三省135县宜居性评价_总得分.xlsx")

# ====================== 2. 可视化绘图 ======================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1. 柱状图：展示均值
sns.barplot(x=stats_df.index, y=stats_df['均值'], ax=axes[0], palette='viridis')
axes[0].set_title("各省宜居性综合得分均值对比")
axes[0].set_ylabel("得分均值")

# 2. 箱线图：展示分布与极值 (箱线图最能直观反映内部离散程度)
sns.boxplot(x="省名", y="综合得分", data=df, ax=axes[1], palette='coolwarm')
axes[1].set_title("各省宜居性得分箱线图分布")
axes[1].set_ylabel("综合得分")

plt.tight_layout()
plt.savefig("宜居性统计可视化分析.png", dpi=300)
plt.show()

print("\n✅ 统计分析完成：")
print("1. 统计报表已保存为：三省宜居性统计指标分析.xlsx")
print("2. 可视化图表已保存为：宜居性统计可视化分析.png")









import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ----------------- 1. 环境配置 -----------------
# 设置中文字体（确保图表显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
sns.set_context("notebook", font_scale=1.1)

# ----------------- 2. 数据读取与准备 -----------------
# 注意：这里使用 r"" 来防止 Windows 路径反斜杠转义
input_path = r"D:\桌面应用\三省135县宜居性评价_总得分.xlsx"

if not os.path.exists(input_path):
    print(f"❌ 错误：在路径 {input_path} 下找不到文件，请确认路径是否正确。")
    # 这里为了演示，我们模拟一点数据，你实际运行时请确保文件路径正确
    df = pd.DataFrame({
        '省名': ['江苏省']*45 + ['浙江省']*45 + ['广东省']*45,
        '综合得分': np.random.normal(85, 5, 135)
    })
else:
    df = pd.read_excel(input_path)
    print("✅ 原始数据加载成功！")

# ----------------- 3. 统计分析计算 -----------------
# 使用 reset_index() 让省名回到列中，方便绘图
stats_df = df.groupby("省名")["综合得分"].agg(['mean', 'std', 'max', 'min']).rename(
    columns={'mean': '均值', 'std': '标准差', 'max': '最高分', 'min': '最低分'}
).reset_index()

# 按照均值降序排列，图表会更好看
stats_df = stats_df.sort_values(by='均值', ascending=False)

print("\n📊 统计指标结果：")
print(stats_df)

# 保存统计报表
output_excel = "三省宜居性统计指标分析.xlsx"
stats_df.to_excel(output_excel, index=False)

# ----------------- 4. 高质量可视化绘图 -----------------
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 图 1：柱状图（带误差线）
# ci='sd' 表示显示标准差，能看出各省内部的差异大小
sns.barplot(x='省名', y='综合得分', data=df, ax=axes[0], 
            palette='viridis', capsize=.1, errorbar='sd', order=stats_df['省名'])
axes[0].set_title("各省宜居性综合得分均值（误差线代表标准差）", fontsize=14)
axes[0].set_ylabel("得分", fontsize=12)
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# 图 2：箱线图（更直观展示离散程度和极值）
sns.boxplot(x="省名", y="综合得分", data=df, ax=axes[1], 
            palette='Set3', order=stats_df['省名'])
# 在箱线图上叠加散点，能看到具体每个县的分布情况
sns.stripplot(x="省名", y="综合得分", data=df, ax=axes[1], 
              color=".3", size=4, alpha=0.5, order=stats_df['省名'])
axes[1].set_title("各省宜居性得分分布细节（箱线+散点）", fontsize=14)
axes[1].set_ylabel("得分", fontsize=12)

plt.tight_layout()
plt.savefig("宜居性统计可视化分析.png", dpi=300)
plt.show()

print(f"\n✅ 分析完成！\n1. 统计表：{output_excel}\n2. 可视化图：宜居性统计可视化分析.png")




















import pandas as pd
import numpy as np

# ... (数据读取与预处理代码同前，此处省略) ...

# 存储各维度的得分表
dimension_scores = pd.DataFrame(index=df.index)
final_weights_series = pd.Series(dtype=float)

# ====================== 维度内计算逻辑 ======================
for dim, cols in dimension_map.items():
    # 1. 提取该维度对应的标准化数据
    dim_data = df_scaled[cols]
    
    # 2. 计算该维度内部的权重 (熵值法)
    # 此时计算的是指标在该维度内的相对权重
    p = dim_data / (dim_data.sum(axis=0) + 1e-12)
    n = len(dim_data)
    k = 1.0 / np.log(n)
    entropy = -k * (p * np.log(p + 1e-12)).sum(axis=0)
    d = 1 - entropy
    internal_weights = d / (d.sum() + 1e-12)
    
    # 3. 计算该维度得分 (该县在该维度下的表现)
    # 得分 = 标准化数据 * 维度内权重
    dimension_scores[f"{dim}_得分"] = dim_data.dot(internal_weights)
    
    # 4. 记录用于后续总分计算的权重 (每维度占比 20%)
    final_weights_series = pd.concat([final_weights_series, internal_weights * 0.2])

# ====================== 最终综合得分 ======================
# 最终权重平滑
w_equal = 1.0 / len(all_features)
final_weights_smoothed = 0.8 * final_weights_series + 0.2 * w_equal

# 计算最终总分
df["综合得分"] = df_scaled[all_features].dot(final_weights_smoothed)

# 将维度得分拼接到总表中
df = pd.concat([df, dimension_scores], axis=1)

# 保存带维度得分的结果
df.to_excel("三省宜居性_含维度得分.xlsx", index=False)































































